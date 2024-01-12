from gymnasium import spaces
import numpy as np
import torch as th
import meshcat
import scipy.spatial.transform as transfm

class GPUQuadcopter:
	"""Custom Environment that follows gym interface"""
	metadata = {'render_modes': ['human']}

	def __init__(self, device='cpu', num_envs=1, sys=dict(), dt=1e-3, normalized_actions=True, normalized_observations=True, alpha_cost=1., alpha_terminal_cost=1.):
		# super(Quadcopter, self).__init__()
		# Define model paramters
		m = 0.5
		g = 9.81
		sys_ = {'m': m, 'I': np.diag([4.86*1e-3, 4.86*1e-3, 8.8*1e-3]), 'l': 0.225, 'g': g, 'bk': 1.14*1e-7/(2.98*1e-6),\
				# 'Q': 0.85*np.diag([5, 0.001, 0.001, 5, 0.5, 0.5, 0.05, 0.075, 0.075, 0.05]), 'R': 0.085*np.diag([0.002, 0.001, 0.001, 0.004]),\
				'Q': np.diag([1, 0, 0, 1, 1, 1, 0.01, 0.1, 0.1, 0.01]), 'R': 0.02*np.diag([0.002, 0.001, 0.001, 0.004]),\
				'QT': np.eye(10),\
				'goal': np.array([[1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]), 'u0': np.array([[m*g], [0.], [0.], [0.]]),\
				'T': 3, 'dt': 1e-3, 'lambda_': 1, 'X_DIMS': 12, 'U_DIMS': 4,\
				'x_sample_limits': np.array([[-2., 2.], [-2., 2.], [0.6, 1.4], [-np.pi/5, np.pi/5], [-np.pi/5, np.pi/5], [-2*np.pi/5, 2*np.pi/5], [-3., 3.], [-3., 3.], [-3., 3.], [-3., 3.], [-3., 3.], [-3., 3.]]),\
				'x_bounds': np.array([[-10., 10.], [-10., 10.], [0., 2.], [-2*np.pi/3, 2*np.pi/3], [-2*np.pi/3, 2*np.pi/3], [-2*np.pi, 2*np.pi], [-12., 12.], [-12., 12.], [-12., 12.], [-12., 12.], [-12., 12.], [-12., 12.]]),\
				'u_limits': np.array([[0, 2*m*g], [-0.35*m*g, 0.35*m*g], [-0.35*m*g, 0.35*m*g], [-0.175*m*g, 0.175*m*g]])
				}
		sys_.update(sys)
		sys_.update({'dt':dt})
		sys_['gamma_'] = np.exp(-sys_['lambda_']*sys_['dt'])
		sys.update(sys_)
		self.device = device
		self.np_dtype = np.float32
		self.th_dtype = th.float32
		self.num_envs = num_envs

		self.X_DIMS = sys['X_DIMS'] # dimension of observations
		self.independent_sampling_dims = np.arange(self.X_DIMS)
		self.observation_dims = np.arange(10) + 2 # drop the xy co-ordinate in the observation
		self.U_DIMS = sys['U_DIMS']
		self.goal = sys['goal']
		self.u0 = sys['u0']
		self.T = sys['T']
		self.dt = sys['dt']
		self.horizon = round(self.T / self.dt)
		self.x_sample_limits = sys['x_sample_limits'] # reset within these limits
		self.x_bounds = sys['x_bounds']
		self.u_limits = sys['u_limits']
		self.Q = sys['Q']
		self.QT = sys['QT']
		self.R = sys['R']
		self.gamma_ = sys['gamma_']
		self.lambda_ = sys['lambda_']

		self.m = sys['m']
		self.l = sys['l']
		self.g = sys['g']
		self.bk = sys['bk']
		self.I = sys['I']
		self.normalized_actions = normalized_actions
		self.normalized_observations = normalized_observations
		self.alpha_cost = alpha_cost
		self.alpha_action_cost = alpha_cost
		self.alpha_terminal_cost = alpha_terminal_cost
		# self.reset()

		# Define action and observation space
		# They must be gym.spaces objects
		# Example when using continuous actions:
		if (normalized_actions):
			self.action_space = spaces.Box(low=-1, high=1, shape=(self.U_DIMS,), dtype=self.np_dtype)
		else:
			self.action_space = spaces.Box(low=self.u_limits[:,0], high=self.u_limits[:,1], dtype=self.np_dtype)

		state_obs_bounds_mid = 0.5*(self.x_bounds[self.observation_dims,1] + self.x_bounds[self.observation_dims,0])
		state_obs_bounds_range = 0.5*(self.x_bounds[self.observation_dims,1] - self.x_bounds[self.observation_dims,0])
		time_obs_bounds_mid = 0.5*self.horizon
		time_obs_bounds_range = 0.5*self.horizon
		obs_bounds_mid = np.concatenate((state_obs_bounds_mid, np.array([time_obs_bounds_mid])))
		obs_bounds_range = np.concatenate((state_obs_bounds_range, np.array([time_obs_bounds_range])))			
		
		if (normalized_observations):
			self.observation_space = spaces.Box(low=-1, high=1, shape=(self.observation_dims.shape[0]+1,), dtype=self.np_dtype)
			target_obs_mid = np.zeros(self.observation_dims.shape[0]+1)
			target_obs_range = np.ones(self.observation_dims.shape[0]+1)
		else:
			self.observation_space = spaces.Box(
				low=np.concatenate((self.x_bounds[self.observation_dims,0], np.array([0.]))), 
				high=np.concatenate((self.x_bounds[self.observation_dims,1], np.array([self.horizon]))), 
				dtype=self.np_dtype
			)
			target_obs_mid = obs_bounds_mid
			target_obs_range = obs_bounds_range
		
		# Create tensor copies of relevant numpy arrays
		self.th_x_bounds = th.asarray(sys['x_bounds'], device=device, dtype=self.th_dtype)
		self.th_u_limits = th.asarray(sys['u_limits'], device=device, dtype=self.th_dtype)
		self.th_Q = th.asarray(sys['Q'], device=device, dtype=self.th_dtype)
		self.th_QT = th.asarray(sys['QT'], device=device, dtype=self.th_dtype)
		self.th_goal = th.asarray(sys['goal'][:,0], device=device, dtype=self.th_dtype)
		self.th_R = th.asarray(sys['R'], device=device, dtype=self.th_dtype)
		self.th_u0 = th.asarray(sys['u0'][:,0], device=device, dtype=self.th_dtype)

		self.th_x_sample_limits_mid = th.asarray(0.5*(sys['x_sample_limits'][:,0] + sys['x_sample_limits'][:,1]), device=device, dtype=self.th_dtype)
		self.th_x_sample_limits_range = th.asarray((sys['x_sample_limits'][:,1] - sys['x_sample_limits'][:,0]), device=device, dtype=self.th_dtype)
		self.th_obs_bounds_mid = th.asarray(obs_bounds_mid, device=device, dtype=self.th_dtype)
		self.th_obs_bounds_range = th.asarray(obs_bounds_range, device=device, dtype=self.th_dtype)
		self.th_target_obs_mid = th.asarray(target_obs_mid, device=device, dtype=self.th_dtype)
		self.th_target_obs_range = th.asarray(target_obs_range, device=device, dtype=self.th_dtype)

		# initialize torch tensors
		self.set_num_envs(num_envs)

	def set_num_envs(self, num_envs):
		self.num_envs = num_envs
		self.state = th.zeros((self.num_envs, self.X_DIMS), device=self.device, dtype=self.th_dtype)
		self.step_count = th.zeros((self.num_envs, ), device=self.device, dtype=self.th_dtype)
		self.obs = th.zeros((self.num_envs, self.observation_dims.shape[0]+1), device=self.device, dtype=self.th_dtype)
		self.cumm_reward = th.zeros((self.num_envs, ), device=self.device, dtype=self.th_dtype)

	def step(self, action):
		# If scaling actions use this
		if (self.normalized_actions):
			action = 0.5*((self.th_u_limits[:,0] + self.th_u_limits[:,1]) + action*(self.th_u_limits[:,1] - self.th_u_limits[:,0]))

		state_ = self.dyn_rk4(self.state, action, self.dt)
		state_ = th.minimum(self.th_x_bounds[:,1], th.maximum(self.th_x_bounds[:,0], state_))

		cost, reached_goal = self._get_cost(action, state_)

		self.state = state_
		self.step_count += 1.
		self._set_obs()
		done = th.logical_or(self.step_count >= self.horizon, reached_goal)
		terminal_cost = self._get_terminal_cost(done=done)
		cost += terminal_cost
		reward = -cost
		self.cumm_reward += reward

		if (th.any(done)):
			info = {
				"ep_reward": th.nanmean(self.cumm_reward[done]).cpu().numpy(),
				"ep_length": th.nanmean(self.step_count[done]).cpu().numpy(),
				"ep_terminal_goal_dist": th.nanmean(self.get_goal_dist()[done]).cpu().numpy(),
				"ep_terminal_cost": th.nanmean(terminal_cost[done]).cpu().numpy()
			}
		else:
			info = {}

		return self.get_obs(), reward, done, False, info

	def reset(self, done=None, seed=None, options=None, state=None):
		if (state is not None):
			assert (len(state.shape)==2 and state.shape[0]==self.num_envs and state.shape[1]==self.X_DIMS), 'Invalid input state'
			self.state[:] = state
			self.step_count[:] = 0.
			self.cumm_reward[:] = 0.

		else:
			if (done is not None):
				assert done.shape[0]==self.num_envs and type(done)==th.Tensor and done.dtype==th.bool, "done tensor of shape %d and type %s"%(done.shape[0], type(done))
			else:
				done = th.ones(self.num_envs, device=self.device, dtype=bool)
			num_done = th.sum(done)
			if (num_done > 0):
				step_count_new = 0*th.randint(low=0, high=self.horizon, size=(num_done,1), dtype=th.float32, device=self.device)
				state_new = ((th.rand((num_done, self.independent_sampling_dims.shape[0]), device=self.device, dtype=self.th_dtype) - 0.5) * (self.th_x_sample_limits_range))
				state_new = (((self.horizon - step_count_new) / self.horizon) * state_new)
				state_new += self.th_x_sample_limits_mid 

				self.step_count[done] = step_count_new[:,0]
				self.state[done,:] = state_new
				self.cumm_reward[done] = 0.

		self._set_obs()
		info = {}
		
		return self.get_obs(), info

	def get_obs(self, normalized=None):
		obs = self.obs
		if ((normalized == None) and self.normalized_observations) or (normalized == True):
			obs = (obs - self.th_obs_bounds_mid) / self.th_obs_bounds_range
			obs = self.th_target_obs_range*obs + self.th_target_obs_mid
		return obs
	
	def get_goal_dist(self):
		return th.linalg.norm((self.state[:,self.observation_dims] - self.th_goal), dim=1, keepdim=False)

	def _set_obs(self):
		self.obs[:,:self.observation_dims.shape[0]] = self.state[:,self.observation_dims]
		self.obs[:,-1] = self.step_count

	def _get_cost(self, action, state_):
		x = self.obs[:,:self.observation_dims.shape[0]]

		cost = th.sum(((x - self.th_goal) @ self.th_Q) * (x - self.th_goal), dim=1, keepdim=False) * self.alpha_cost 
		cost = cost + th.sum(((action - self.th_u0) @ self.th_R) * (action - self.th_u0), dim=1, keepdim=False) * self.alpha_action_cost
		cost = cost * self.dt

		reached_goal = th.linalg.norm(state_[:,self.observation_dims] - self.th_goal, dim=1, keepdim=False) <= 1e-2	

		return cost, reached_goal
	
	def _get_terminal_cost(self, done=None):
		if (done is not None):
			assert done.shape[0]==self.num_envs and type(done)==th.Tensor and done.dtype==th.bool, "done tensor of shape %d and type %s"%(done.shape[0], type(done))
		else:
			done = th.ones(self.num_envs, device=self.device, dtype=bool)
		x = self.obs[:,:self.observation_dims.shape[0]]
		cost = th.sum(((x - self.th_goal) @ self.th_QT) * (x - self.th_goal), dim=1, keepdim=False) * self.alpha_terminal_cost
		cost = cost * done

		return cost

	def dyn_rk4(self, x, u, dt):
		k1 = self.dyn_full(x, u)
		q = x + 0.5*k1*dt

		k2 = self.dyn_full(q, u)
		q = x + 0.5*k2*dt

		k3 = self.dyn_full(q, u)
		q = x + k3*dt

		k4 = self.dyn_full(q, u)
		
		q = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

		return q
	
	def dyn_full(self, x, u):

		m = self.m
		l = self.l
		g = self.g
		bk = self.bk
		II = self.I
		I1 = II[0,0]
		I2 = II[1,1]
		I3 = II[2,2]

		x2 = x[:,3]
		x3 = x[:,4]
		x4 = x[:,5]
		x7 = x[:,8]
		x8 = x[:,9]
		x9 = x[:,10]
		x10 = x[:,11]

		u1 = u[:,0]
		u2 = u[:,1]
		u3 = u[:,2]
		u4 = u[:,3]

		t2 = th.cos(x2)
		t3 = th.cos(x3)
		t4 = th.cos(x4)
		t5 = th.sin(x2)
		t6 = th.sin(x3)
		t7 = th.sin(x4)
		t8 = I1**2
		t9 = I2**2
		t10 = I3**2
		t11 = x2*2.0
		t12 = x3*2.0
		t13 = x9**2
		t14 = x10**2
		t20 = 1.0/I2
		t21 = 1.0/I3
		t22 = 1.0/m
		t15 = t2**2
		t16 = t3**2
		t17 = t3**3
		t18 = th.sin(t11)
		t19 = th.sin(t12)
		t23 = 1.0/t3
		et1 = I1*t10*x9*x10-I3*t8*x9*x10+I1*I2*I3*x9*x10+I2*I3*l*t3*u2-I1*t6*t10*x8*x9+I3*t6*t8*x8*x9+I1*t9*t15*x9*x10-I2*t8*t15*x9*x10-I1*t10*t15*x9*x10+I3*t8*t15*x9*x10-I1*t10*t16*x9*x10+I3*t8*t16*x9*x10+I2*t10*t16*x9*x10-I3*t9*t16*x9*x10+I1*I2*I3*t6*x8*x9+I1*I2*bk*t2*t6*u4+I1*I3*l*t5*t6*u3+I1*t2*t3*t5*t9*t14-I2*t2*t3*t5*t8*t14-I1*t2*t3*t5*t10*t14+I2*t2*t3*t5*t10*t13+I3*t2*t3*t5*t8*t14-I3*t2*t3*t5*t9*t13-I1*t2*t5*t9*t14*t17+I2*t2*t5*t8*t14*t17+I1*t2*t5*t10*t14*t17-I3*t2*t5*t8*t14*t17
		et2 = -I2*t2*t5*t10*t14*t17+I3*t2*t5*t9*t14*t17-I1*t6*t9*t15*x8*x9+I2*t6*t8*t15*x8*x9+I1*t6*t10*t15*x8*x9-I3*t6*t8*t15*x8*x9-I1*t9*t15*t16*x9*x10+I2*t8*t15*t16*x9*x10+I1*t10*t15*t16*x9*x10-I3*t8*t15*t16*x9*x10-I2*t10*t15*t16*x9*x10*2.0+I3*t9*t15*t16*x9*x10*2.0-I1*t2*t3*t5*t6*t9*x8*x10+I2*t2*t3*t5*t6*t8*x8*x10+I1*t2*t3*t5*t6*t10*x8*x10-I3*t2*t3*t5*t6*t8*x8*x10

		dx = th.zeros((self.num_envs, self.X_DIMS), device=self.device)
		dx[:, 0:6] = x[:,6:12]
		dx[:,6] = t22*u1*(t5*t7+t2*t4*t6)
		dx[:,7] = -t22*u1*(t4*t5-t2*t6*t7)
		dx[:,8] = -g+t2*t3*t22*u1
		dx[:,9] = (t20*t21*t23*(et1+et2))/I1
		dx[:,10] = t20*t21*(t9*t14*t19-t3*t9*x8*x10*2.0-t9*t18*x8*x9+t10*t18*x8*x9-I1*I2*t14*t19+I2*bk*t5*u4*2.0-I3*l*t2*u3*2.0+I1*I2*t3*x8*x10*2.0+I2*I3*t3*x8*x10*2.0+I1*I2*t18*x8*x9-I1*I3*t18*x8*x9-t3*t6*t9*t14*t15*2.0+t3*t6*t10*t14*t15*2.0+t3*t9*t15*x8*x10*2.0-t3*t10*t15*x8*x10*2.0+t2*t5*t6*t9*x9*x10*2.0-t2*t5*t6*t10*x9*x10*2.0+I1*I2*t3*t6*t14*t15*2.0-I1*I3*t3*t6*t14*t15*2.0-I1*I2*t3*t15*x8*x10*2.0+I1*I3*t3*t15*x8*x10*2.0-I1*I2*t2*t5*t6*x9*x10*2.0+I1*I3*t2*t5*t6*x9*x10*2.0)*(-1.0/2.0)
		dx[:,11] = t20*t21*t23*(-t10*x8*x9+t6*t10*x9*x10-t9*t15*x8*x9+t10*t15*x8*x9+I1*I3*x8*x9+I2*I3*x8*x9+I2*bk*t2*u4+I3*l*t5*u3-I1*I3*t6*x9*x10+I2*I3*t6*x9*x10+I1*I2*t15*x8*x9-I1*I3*t15*x8*x9+t6*t9*t15*x9*x10-t6*t10*t15*x9*x10+t2*t3*t5*t6*t9*t14-t2*t3*t5*t6*t10*t14-t2*t3*t5*t9*x8*x10+t2*t3*t5*t10*x8*x10-I1*I2*t6*t15*x9*x10+I1*I3*t6*t15*x9*x10-I1*I2*t2*t3*t5*t6*t14+I1*I3*t2*t3*t5*t6*t14+I1*I2*t2*t3*t5*x8*x10-I1*I3*t2*t3*t5*x8*x10)

		return dx

	def _create_visualizer(self):
		#TODO visualize all the agents in parallel
		self.viz = meshcat.Visualizer()
		# Create the quadcopter geometry
		self.viz['root'].set_object(meshcat.geometry.Box([2*self.l, 0.01, 0.01]))  # units in meters
		self.viz['root']['wing'].set_object(meshcat.geometry.Box([0.01, 2*self.l, 0.01]))
		
		motor_color = 0x505050
		motor_reflectivity = 0.9

		self.viz['root']['motor1'].set_object(
			meshcat.geometry.Cylinder(height=0.06, radius=0.03),
			meshcat.geometry.MeshLambertMaterial(color=motor_color, reflectivity=motor_reflectivity))
		poseC1 = np.eye(4)
		poseC1[:3,:3] = transfm.Rotation.from_euler('yxz', [0., np.pi/2, 0.]).as_matrix()
		poseC1[:3,3] = np.array([self.l, 0., 0.])
		self.viz['root']['motor1'].set_transform(poseC1)

		self.viz['root']['motor2'].set_object(
			meshcat.geometry.Cylinder(height=0.06, radius=0.03),
			meshcat.geometry.MeshLambertMaterial(color=motor_color, reflectivity=motor_reflectivity))
		poseC2 = np.eye(4)
		poseC2[:3,:3] = transfm.Rotation.from_euler('yxz', [0., np.pi/2, 0.]).as_matrix()
		poseC2[:3,3] = np.array([0., self.l, 0.])
		self.viz['root']['motor2'].set_transform(poseC2)

		self.viz['root']['motor3'].set_object(
			meshcat.geometry.Cylinder(height=0.06, radius=0.03),
			meshcat.geometry.MeshLambertMaterial(color=motor_color, reflectivity=motor_reflectivity))
		poseC3 = np.eye(4)
		poseC3[:3,:3] = transfm.Rotation.from_euler('yxz', [0., np.pi/2, 0.]).as_matrix()
		poseC3[:3,3] = np.array([-self.l, 0., 0.])
		self.viz['root']['motor3'].set_transform(poseC3)

		self.viz['root']['motor4'].set_object(
			meshcat.geometry.Cylinder(height=0.06, radius=0.03),
			meshcat.geometry.MeshLambertMaterial(color=motor_color, reflectivity=motor_reflectivity))
		poseC4 = np.eye(4)
		poseC4[:3,:3] = transfm.Rotation.from_euler('yxz', [0., np.pi/2, 0.]).as_matrix()
		poseC4[:3,3] = np.array([0., -self.l, 0.])
		self.viz['root']['motor4'].set_transform(poseC4)

		# self.viz['root']['pendulum'].set_object(
		# 	meshcat.geometry.Box([0.01, 0.01, 0.9]),
		# 	meshcat.geometry.MeshLambertMaterial(color=motor_color, reflectivity=motor_reflectivity))
		# poseP = np.eye(4)
		# poseP[:3,3] = np.array([0., 0., 0.45])
		# self.viz['root']['pendulum'].set_transform(poseP)

		heading_color = 0x880808
		heading_reflectivity = 0.95
		self.viz['root']['heading'].set_object(
			meshcat.geometry.TriangularMeshGeometry(
				vertices=np.array([[0., self.l/3, 0.], [0., -self.l/3, 0.], [0.9*self.l, 0., 0.]]),
				faces=np.array([[0, 1, 2]])
			),
			meshcat.geometry.MeshLambertMaterial(color=heading_color, reflectivity=heading_reflectivity)
		)

	def render(self, mode='human'):
		if (not (hasattr(self, 'viz') and isinstance(self.viz, meshcat.Visualizer))):
			self._create_visualizer()
		pose = np.eye(4)
		pose[:3,3] = self.state[:3]
		pose[:3,:3] = transfm.Rotation.from_euler('yxz', [self.state[4], self.state[3], self.state[5]]).as_matrix()
		self.viz['root'].set_transform(pose)

	def close (self):
		pass