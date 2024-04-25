from gymnasium import spaces
import numpy as np
import torch as th
import meshcat
import scipy.spatial.transform as transfm

class GPUQuadcopterDecomposition:
	"""Custom Environment that follows gym interface"""
	metadata = {'render_modes': ['human']}

	def __init__(self, device='cpu', num_envs=1, param=dict(), normalized_actions=True, normalized_observations=True, alpha_cost=1., alpha_terminal_cost=1.):
		# super(Quadcopter, self).__init__()
		# Define model paramters
		m = 0.5
		g = 9.81
		param_ = {'m': m, 'I': np.diag([4.86*1e-3, 4.86*1e-3, 8.8*1e-3]), 'l': 0.225, 'g': g, 'bk': 1.14*1e-7/(2.98*1e-6),\
				# 'Q': 0.85*np.diag([5, 0.001, 0.001, 5, 0.5, 0.5, 0.05, 0.075, 0.075, 0.05]), 'R': 0.085*np.diag([0.002, 0.001, 0.001, 0.004]),\
				'Q': np.diag([0, 0, 1, 0, 0, 1, 1, 1, 0.01, 0.1, 0.1, 0.01]), 'R': 0.02*np.diag([0.002, 0.001, 0.001, 0.004]),\
				'QT': np.diag(np.concatenate((np.zeros(2), np.ones(10)))),\
				'goal': np.array([[0.], [0.], [1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]), 'u0': np.array([[m*g], [0.], [0.], [0.]]),\
				'T': 3, 'dt': 1e-3, 'lambda_': 1, 'X_DIMS': 12, 'U_DIMS': 4,\
				'X_DIMS_FREE': np.arange(11)+2, 'X_DIMS_FIXED': np.array([]), 'U_DIMS_FREE': np.arange(4), 'U_DIMS_FIXED': np.array([]), 'U_DIMS_CONTROLLED': np.array([]),\
				'x_sample_limits': np.array([[-2., 2.], [-2., 2.], [0.6, 1.4], [-np.pi/5, np.pi/5], [-np.pi/5, np.pi/5], [-2*np.pi/5, 2*np.pi/5], [-3., 3.], [-3., 3.], [-3., 3.], [-3., 3.], [-3., 3.], [-3., 3.]]),\
				'x_bounds': np.array([[-10., 10.], [-10., 10.], [0., 2.], [-2*np.pi/3, 2*np.pi/3], [-2*np.pi/3, 2*np.pi/3], [-2*np.pi, 2*np.pi], [-12., 12.], [-12., 12.], [-12., 12.], [-12., 12.], [-12., 12.], [-12., 12.]]),\
				'u_limits': np.array([[0, 2*m*g], [-0.35*m*g, 0.35*m*g], [-0.35*m*g, 0.35*m*g], [-0.7*m*g, 0.7*m*g]])
				}
		param_.update(param)
		param_['gamma_'] = np.exp(-param_['lambda_']*param_['dt'])
		param.update(param_)
		self.device = device
		self.np_dtype = np.float32
		self.th_dtype = th.float32
		self.num_envs = num_envs

		self.X_DIMS = param['X_DIMS'] # dimension of observations
		self.X_DIMS_FREE = param['X_DIMS_FREE']
		self.X_DIMS_FIXED = param['X_DIMS_FIXED']
		self.independent_sampling_dims = np.arange(10) + 2
		# check that X_DIMS_FIXED and X_DIMS_FREE are subsets of independent_sampling_dims
		assert np.all(np.any(self.X_DIMS_FREE[:,np.newaxis]==np.concatenate((self.independent_sampling_dims, np.array([self.X_DIMS]))), axis=1)), 'free dims must be a subset of the independent dims'
		assert np.all(np.any(self.X_DIMS_FIXED[:,np.newaxis]==np.concatenate((self.independent_sampling_dims, np.array([self.X_DIMS]))), axis=1)), 'fixed dims must be a subset of the independent dims'
		self.observation_dims = np.arange(self.X_DIMS)
		self.cost_dims_free = np.arange(10) + 2
		self.cost_dims_free = self.cost_dims_free[np.any(self.cost_dims_free[:,np.newaxis] == self.X_DIMS_FREE, axis=1)]

		self.U_DIMS = param['U_DIMS']
		self.U_DIMS_FREE = param['U_DIMS_FREE']
		self.U_DIMS_FIXED = param['U_DIMS_FIXED']
		self.U_DIMS_CONTROLLED = param['U_DIMS_CONTROLLED']

		self.goal = param['goal']
		self.u0 = param['u0']
		self.T = param['T']
		self.dt = param['dt']
		self.horizon = round(self.T / self.dt)
		self.x_sample_limits = param['x_sample_limits'][self.independent_sampling_dims,:] # reset within these limits
		self.x_bounds = param['x_bounds']
		self.u_limits = param['u_limits']
		self.Q = param['Q']
		self.QT = param['QT']
		self.R = param['R']
		self.gamma_ = param['gamma_']
		self.lambda_ = param['lambda_']

		self.m = param['m']
		self.l = param['l']
		self.g = param['g']
		self.bk = param['bk']
		self.I = param['I']
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
			self.action_space = spaces.Box(low=-1, high=1, shape=(self.U_DIMS,), dtype=np.float32)
		else:
			self.action_space = spaces.Box(low=self.u_limits[:,0], high=self.u_limits[:,1], dtype=np.float32)

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
		self.th_x_bounds = th.asarray(self.x_bounds, device=device, dtype=self.th_dtype)
		self.th_u_limits = th.asarray(self.u_limits, device=device, dtype=self.th_dtype)
		self.th_Q = th.asarray(self.Q, device=device, dtype=self.th_dtype)
		self.th_QT = th.asarray(self.QT, device=device, dtype=self.th_dtype)
		self.th_goal = th.asarray(self.goal[:,0], device=device, dtype=self.th_dtype)
		self.th_R = th.asarray(self.R, device=device, dtype=self.th_dtype)
		self.th_u0 = th.asarray(self.u0[:,0], device=device, dtype=self.th_dtype)

		self.th_action_space_low = th.asarray(self.action_space.low, device=device, dtype=self.th_dtype)
		self.th_action_space_high = th.asarray(self.action_space.low, device=device, dtype=self.th_dtype)
		self.th_x_sample_limits_mid = th.asarray(0.5*(self.x_sample_limits[:,0] + self.x_sample_limits[:,1]), device=device, dtype=self.th_dtype)
		self.th_x_sample_limits_range = th.asarray((self.x_sample_limits[:,1] - self.x_sample_limits[:,0]), device=device, dtype=self.th_dtype)
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
		action[:,self.U_DIMS_FIXED] = self.th_u0[self.U_DIMS_FIXED] #TODO should it be 0 instead?

		state_ = self.dyn_rk4(self.state, action, self.dt)
		state_ = th.minimum(self.th_x_bounds[:,1], th.maximum(self.th_x_bounds[:,0], state_))

		cost, reached_goal = self._get_cost(action, state_)
		reward = -cost

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
			state[:,self.X_DIMS_FIXED] = self.th_goal[self.X_DIMS_FIXED]
			self.state[:] = state
			self.step_count[:] = 0.
			self.cumm_reward[:] = 0.

		else:
			if (done is not None):
				assert done.shape[0]==self.num_envs and type(done)==th.Tensor and done.dtype==th.bool, "done tensor of shape %d and type %s"%(done.shape[0], type(done))
			else:
				done = th.ones(self.num_envs, device=self.device, dtype=th.bool)
			num_done = th.sum(done)
			if (num_done > 0):
				# step_count_new = th.randint(low=0, high=self.horizon, size=(num_done,1), dtype=th.float32, device=self.device)
				step_count_new = th.zeros((num_done, 1), device=self.device, dtype=self.th_dtype)
				state_new = ((th.rand((num_done, self.independent_sampling_dims.shape[0]), device=self.device, dtype=self.th_dtype) - 0.5) * (self.th_x_sample_limits_range))
				state_new = (((self.horizon - step_count_new) / self.horizon) * state_new)
				state_new += th.unsqueeze(self.th_x_sample_limits_mid, dim=0) 
				state = th.zeros(num_done, self.X_DIMS, device=self.device, dtype=self.th_dtype)
				state[:,self.independent_sampling_dims] = state_new
				state[:,self.X_DIMS_FIXED] = self.th_goal[self.X_DIMS_FIXED]

				self.step_count[done] = step_count_new[:,0]
				self.state[done,:] = state
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
		return th.linalg.norm((self.state[:,self.cost_dims_free] - self.th_goal[self.cost_dims_free]), dim=1, keepdim=False)

	def _set_obs(self):
		self.obs[:,:self.observation_dims.shape[0]] = self.state[:,self.observation_dims]
		self.obs[:,-1] = self.step_count

	def _get_cost(self, action, state_):
		y = self.obs[:,:self.observation_dims.shape[0]] - self.th_goal[self.observation_dims]

		cost = th.sum((y @ self.th_Q) * y, dim=1, keepdim=False) * self.alpha_cost 
		cost = cost + th.sum(((action - self.th_u0) @ self.th_R) * (action - self.th_u0), dim=1, keepdim=False) * self.alpha_action_cost
		cost = cost * self.dt

		reached_goal = th.linalg.norm(state_[:,self.observation_dims] - self.th_goal[self.observation_dims], dim=1, keepdim=False) <= 1e-2	

		return cost, reached_goal
	
	def _get_terminal_cost(self, done=None):
		if (done is not None):
			assert done.shape[0]==self.num_envs and type(done)==th.Tensor and done.dtype==th.bool, "done tensor of shape %d and type %s"%(done.shape[0], type(done))
		else:
			done = th.ones(self.num_envs, device=self.device, dtype=th.bool)
		y = self.obs[:,:self.observation_dims.shape[0]] - self.th_goal[self.observation_dims]
		cost = th.sum((y @ self.th_QT) * y, dim=1, keepdim=False) * self.alpha_terminal_cost
		cost = cost * done

		return cost
	
	def dyn_rk4(self, x, u, dt):
		k1 = self.dyn_full(x, u)
		k1[:,self.X_DIMS_FIXED] = 0.
		q = x + 0.5*k1*dt

		k2 = self.dyn_full(q, u)
		k2[:,self.X_DIMS_FIXED] = 0.
		q = x + 0.5*k2*dt

		k3 = self.dyn_full(q, u)
		k3[:,self.X_DIMS_FIXED] = 0.
		q = x + k3*dt

		k4 = self.dyn_full(q, u)
		k4[:,self.X_DIMS_FIXED] = 0.
		
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

		dx = th.zeros((self.num_envs, self.X_DIMS), device=self.device, dtype=self.th_dtype)
		dx[:, 0:6] = x[:,6:12]
		dx[:,6] = t22*u1*(t5*t7+t2*t4*t6)
		dx[:,7] = -t22*u1*(t4*t5-t2*t6*t7)
		dx[:,8] = -g+t2*t3*t22*u1
		dx[:,9] = (t20*t21*t23*(et1+et2))/I1
		dx[:,10] = t20*t21*(t9*t14*t19-t3*t9*x8*x10*2.0-t9*t18*x8*x9+t10*t18*x8*x9-I1*I2*t14*t19+I2*bk*t5*u4*2.0-I3*l*t2*u3*2.0+I1*I2*t3*x8*x10*2.0+I2*I3*t3*x8*x10*2.0+I1*I2*t18*x8*x9-I1*I3*t18*x8*x9-t3*t6*t9*t14*t15*2.0+t3*t6*t10*t14*t15*2.0+t3*t9*t15*x8*x10*2.0-t3*t10*t15*x8*x10*2.0+t2*t5*t6*t9*x9*x10*2.0-t2*t5*t6*t10*x9*x10*2.0+I1*I2*t3*t6*t14*t15*2.0-I1*I3*t3*t6*t14*t15*2.0-I1*I2*t3*t15*x8*x10*2.0+I1*I3*t3*t15*x8*x10*2.0-I1*I2*t2*t5*t6*x9*x10*2.0+I1*I3*t2*t5*t6*x9*x10*2.0)*(-1.0/2.0)
		dx[:,11] = t20*t21*t23*(-t10*x8*x9+t6*t10*x9*x10-t9*t15*x8*x9+t10*t15*x8*x9+I1*I3*x8*x9+I2*I3*x8*x9+I2*bk*t2*u4+I3*l*t5*u3-I1*I3*t6*x9*x10+I2*I3*t6*x9*x10+I1*I2*t15*x8*x9-I1*I3*t15*x8*x9+t6*t9*t15*x9*x10-t6*t10*t15*x9*x10+t2*t3*t5*t6*t9*t14-t2*t3*t5*t6*t10*t14-t2*t3*t5*t9*x8*x10+t2*t3*t5*t10*x8*x10-I1*I2*t6*t15*x9*x10+I1*I3*t6*t15*x9*x10-I1*I2*t2*t3*t5*t6*t14+I1*I3*t2*t3*t5*t6*t14+I1*I2*t2*t3*t5*x8*x10-I1*I3*t2*t3*t5*x8*x10)

		return dx
	
	def _create_visualizer(self):
		self.viz = meshcat.Visualizer()
		for id in range(self.num_envs):
			a_id = 'root' + str(id)
			
			# Create the quadcopter geometry
			self.viz['root'][a_id].set_object(meshcat.geometry.Box([2*self.l, 0.01, 0.01]))  # units in meters
			self.viz['root'][a_id]['wing'].set_object(meshcat.geometry.Box([0.01, 2*self.l, 0.01]))
			
			motor_color = 0x505050
			motor_reflectivity = 0.9

			self.viz['root'][a_id]['motor1'].set_object(
				meshcat.geometry.Cylinder(height=0.06, radius=0.03),
				meshcat.geometry.MeshLambertMaterial(color=motor_color, reflectivity=motor_reflectivity))
			poseC1 = np.eye(4)
			poseC1[:3,:3] = transfm.Rotation.from_euler('yxz', [0., np.pi/2, 0.]).as_matrix()
			poseC1[:3,3] = np.array([self.l, 0., 0.])
			self.viz['root'][a_id]['motor1'].set_transform(poseC1)

			self.viz['root'][a_id]['motor2'].set_object(
				meshcat.geometry.Cylinder(height=0.06, radius=0.03),
				meshcat.geometry.MeshLambertMaterial(color=motor_color, reflectivity=motor_reflectivity))
			poseC2 = np.eye(4)
			poseC2[:3,:3] = transfm.Rotation.from_euler('yxz', [0., np.pi/2, 0.]).as_matrix()
			poseC2[:3,3] = np.array([0., self.l, 0.])
			self.viz['root'][a_id]['motor2'].set_transform(poseC2)

			self.viz['root'][a_id]['motor3'].set_object(
				meshcat.geometry.Cylinder(height=0.06, radius=0.03),
				meshcat.geometry.MeshLambertMaterial(color=motor_color, reflectivity=motor_reflectivity))
			poseC3 = np.eye(4)
			poseC3[:3,:3] = transfm.Rotation.from_euler('yxz', [0., np.pi/2, 0.]).as_matrix()
			poseC3[:3,3] = np.array([-self.l, 0., 0.])
			self.viz['root'][a_id]['motor3'].set_transform(poseC3)

			self.viz['root'][a_id]['motor4'].set_object(
				meshcat.geometry.Cylinder(height=0.06, radius=0.03),
				meshcat.geometry.MeshLambertMaterial(color=motor_color, reflectivity=motor_reflectivity))
			poseC4 = np.eye(4)
			poseC4[:3,:3] = transfm.Rotation.from_euler('yxz', [0., np.pi/2, 0.]).as_matrix()
			poseC4[:3,3] = np.array([0., -self.l, 0.])
			self.viz['root'][a_id]['motor4'].set_transform(poseC4)

			# self.viz['root'][a_id]['pendulum'].set_object(
			# 	meshcat.geometry.Box([0.01, 0.01, 0.9]),
			# 	meshcat.geometry.MeshLambertMaterial(color=motor_color, reflectivity=motor_reflectivity))
			# poseP = np.eye(4)
			# poseP[:3,3] = np.array([0., 0., 0.45])
			# self.viz['root'][a_id]['pendulum'].set_transform(poseP)

			heading_color = 0x880808
			heading_reflectivity = 0.95
			self.viz['root'][a_id]['heading'].set_object(
				meshcat.geometry.TriangularMeshGeometry(
					vertices=np.array([[0., self.l/3, 0.], [0., -self.l/3, 0.], [0.9*self.l, 0., 0.]]),
					faces=np.array([[0, 1, 2]])
				),
				meshcat.geometry.MeshLambertMaterial(color=heading_color, reflectivity=heading_reflectivity)
			)

	def render(self, mode='human'):
		if (not (hasattr(self, 'viz') and isinstance(self.viz, meshcat.Visualizer))):
			self._create_visualizer()
		
		state_ = self.state.cpu().numpy()
		for id in range(self.num_envs):
			a_id = 'root' + str(id)

			pose = np.eye(4)
			pose[:3,3] = state_[id,:3]
			pose[:3,:3] = transfm.Rotation.from_euler('yxz', [state_[id,4], state_[id,3], state_[id,5]]).as_matrix()
			self.viz['root'][a_id].set_transform(pose)
		
		self.viz['root'].set_cam_pos([10*np.cos(np.pi/4 + 2*np.pi/self.horizon*self.step_count[0].cpu().numpy()), 10*np.sin(np.pi/4 + 2*np.pi/self.horizon*self.step_count[0].cpu().numpy()), 3])
		self.viz['root'].set_cam_target([0, 0, 1])
		return self.viz['root'].get_image()

	def close (self):
		pass