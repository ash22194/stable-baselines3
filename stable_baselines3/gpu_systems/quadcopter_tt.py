import os
from gymnasium import spaces
import numpy as np
import torch as th
import meshcat
from scipy.io import loadmat
import scipy.spatial.transform as transfm

class GPUQuadcopterTT:
	"""Custom Environment that follows gym interface"""
	metadata = {'render_modes': ['human']}

	def __init__(self, trajectory_file, device='cpu', num_envs=1, param=dict(), reference_trajectory_horizon=0, normalized_actions=True, normalized_observations=True, alpha_cost=1., alpha_action_cost=1., alpha_terminal_cost=1.):
		# super(Quadcopter, self).__init__()
		# Define model paramters
		m = 0.5
		g = 9.81
		param_ = {'m': m, 'I': np.diag([4.86*1e-3, 4.86*1e-3, 8.8*1e-3]), 'l': 0.225, 'g': g, 'bk': 1.14*1e-7/(2.98*1e-6),
			# 'Q': np.diag([1, 1, 0.1, 0, 0, 0.1, 0.01, 0.01, 0.001, 0.001, 0.001, 0.0001]), 'R': 0.1*np.diag([0.002, 0.001, 0.001, 0.004]), 'QT': 2*np.eye(12),
			'Q': np.diag([1, 1, 1, 0, 0, 1, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]), 'R': 0.1*np.diag([0.002, 0.001, 0.001, 0.004]), 'QT': 2*np.eye(12),
			'u0': np.array([[m*g], [0.], [0.], [0.]]), 'T': 3, 'dt': 1e-3, 'lambda_': 1, 'X_DIMS': 12, 'U_DIMS': 4,
			'x_sample_limits': np.array([[-0.25, 0.25], [-0.25, 0.25], [-0.25, 0.25], [-np.pi/6, np.pi/6], [-np.pi/6, np.pi/6], [-np.pi/6, np.pi/6], [-1.25, 1.25], [-1.25, 1.25], [-1.25, 1.25], [-1.25, 1.25], [-1.25, 1.25], [-1.25, 1.25]]),
			'x_bounds': np.array([[-5., 5.], [-5., 5.], [-5, 5.], [-2*np.pi/3, 2*np.pi/3], [-2*np.pi/3, 2*np.pi/3], [-2*np.pi, 2*np.pi], [-6., 6.], [-6., 6.], [-6., 6.], [-6., 6.], [-6., 6.], [-6., 6.]]),
			'max_thrust_factor': 2
		}
		param_.update(param)
		param_['gamma_'] = np.exp(-param_['lambda_']*param_['dt'])
		param.update(param_)
		self.device = device
		self.np_dtype = np.float32
		self.th_dtype = th.float32
		self.num_envs = num_envs

		self.X_DIMS = param['X_DIMS'] # dimension of observations
		self.independent_sampling_dims = np.arange(self.X_DIMS)
		self.observation_dims = np.arange(self.X_DIMS)
		self.U_DIMS = param['U_DIMS']

		self.u0 = param['u0']
		self.T = param['T']
		self.dt = param['dt']
		self.horizon = round(self.T / self.dt)

		assert trajectory_file.endswith('.mat'), 'Trajectory file must be a .mat'
		trajectory_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'examples/configs/quadcopter_tt/trajectories', trajectory_file)
		trajectory = loadmat(trajectory_file)['trajectory']
		self.reference_trajectory = trajectory.copy()
		self.reference_trajectory_horizon = int(reference_trajectory_horizon/self.T*trajectory.shape[1])

		x_bounds = np.zeros(param['x_bounds'].shape)
		x_bounds[:,0] = param['x_bounds'][:,0] + np.min(trajectory, axis=1) 
		x_bounds[:,1] = param['x_bounds'][:,1] + np.max(trajectory, axis=1) 

		u_limits = np.zeros((self.U_DIMS, 2))
		max_thrust = param['max_thrust_factor']*m*g
		u_limits[0,1] = max_thrust
		u_limits[1,:] = np.array([-0.2*max_thrust, 0.2*max_thrust])
		u_limits[2,:] = np.array([-0.2*max_thrust, 0.2*max_thrust])
		u_limits[3,:] = np.array([-0.4*max_thrust, 0.4*max_thrust])

		self.m = param['m']
		self.l = param['l']
		self.g = param['g']
		self.bk = param['bk']
		self.I = param['I']
		self.normalized_actions = normalized_actions
		self.normalized_observations = normalized_observations
		self.alpha_cost = alpha_cost
		self.alpha_action_cost = alpha_action_cost
		self.alpha_terminal_cost = alpha_terminal_cost

		# Define action and observation space
		# They must be gym.spaces objects
		# Example when using continuous actions:
		if (normalized_actions):
			self.action_space = spaces.Box(low=-1, high=1, shape=(self.U_DIMS,), dtype=np.float32)
		else:
			self.action_space = spaces.Box(low=u_limits[:,0], high=u_limits[:,1], dtype=np.float32)

		state_obs_bounds_mid = 0.5*(x_bounds[self.observation_dims,1] + x_bounds[self.observation_dims,0])
		state_obs_bounds_range = 0.5*(x_bounds[self.observation_dims,1] - x_bounds[self.observation_dims,0])
		time_obs_bounds_mid = 0.5*self.horizon
		time_obs_bounds_range = 0.5*self.horizon
		trajectory_input_obs_bounds_mid = np.tile(0.5*(x_bounds[self.observation_dims,1] + x_bounds[self.observation_dims,0]), self.reference_trajectory_horizon)
		trajectory_input_obs_bounds_range = np.tile(0.5*(x_bounds[self.observation_dims,1] - x_bounds[self.observation_dims,0]), self.reference_trajectory_horizon)
		obs_bounds_mid = np.concatenate((state_obs_bounds_mid, np.array([time_obs_bounds_mid]), trajectory_input_obs_bounds_mid))
		obs_bounds_range = np.concatenate((state_obs_bounds_range, np.array([time_obs_bounds_range]), trajectory_input_obs_bounds_range))			

		print('Observation dimension :', obs_bounds_mid.shape)
		print('X bounds :', x_bounds)

		if (normalized_observations):
			self.observation_space = spaces.Box(low=-1, high=1, shape=obs_bounds_mid.shape, dtype=np.float32)
			target_obs_mid = np.zeros(obs_bounds_mid.shape)
			target_obs_range = np.ones(obs_bounds_range.shape)
		else:
			self.observation_space = spaces.Box(
				low=obs_bounds_mid - obs_bounds_range, 
				high=obs_bounds_mid + obs_bounds_range, 
				dtype=np.float32
			)
			target_obs_mid = obs_bounds_mid
			target_obs_range = obs_bounds_range

		# Create tensor copies of relevant numpy arrays
		self.th_x_bounds = th.asarray(x_bounds, device=device, dtype=self.th_dtype)
		self.th_u_limits = th.asarray(u_limits, device=device, dtype=self.th_dtype)
		self.th_Q = th.asarray(param['Q'], device=device, dtype=self.th_dtype)
		self.th_QT = th.asarray(param['QT'], device=device, dtype=self.th_dtype)
		self.th_R = th.asarray(param['R'], device=device, dtype=self.th_dtype)
		self.th_reference_trajectory = th.asarray(trajectory, device=device, dtype=self.th_dtype)
		self.th_u0 = th.asarray(param['u0'][:,0], device=device, dtype=self.th_dtype)
		self.th_cost_dims = th.arange(self.X_DIMS, device=device, dtype=th.int)
		self.th_tracking_dims = th.asarray([0,1,2,3,4,5], device=device, dtype=th.int)

		self.th_x_sample_limits_mid = th.asarray(0.5*(param['x_sample_limits'][:,0] + param['x_sample_limits'][:,1]), device=device, dtype=self.th_dtype)
		self.th_x_sample_limits_range = th.asarray((param['x_sample_limits'][:,1] - param['x_sample_limits'][:,0]), device=device, dtype=self.th_dtype)
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
		self.obs = th.zeros((self.num_envs, self.th_obs_bounds_mid.shape[0]), device=self.device, dtype=self.th_dtype)
		self.cumm_reward = th.zeros((self.num_envs, ), device=self.device, dtype=self.th_dtype)
		self.tracking_error = th.zeros((self.num_envs, ), device=self.device, dtype=self.th_dtype)

	def step(self, action):
		# If scaling actions use this
		if (self.normalized_actions):
			action = 0.5*((self.th_u_limits[:,0] + self.th_u_limits[:,1]) + action*(self.th_u_limits[:,1] - self.th_u_limits[:,0]))

		state_ = self.dyn_rk4(self.state, action, self.dt)
		state_ = th.minimum(self.th_x_bounds[:,1], th.maximum(self.th_x_bounds[:,0], state_))

		cost, reached_goal = self._get_cost(action, state_)

		self.state = state_
		self.step_count += 1.
		self._update_tracking_error()
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
			self.tracking_error[:] = 0.

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
				state_new += (self._interp_goal(step_count_new) + self.th_x_sample_limits_mid)

				self.step_count[done] = step_count_new[:,0]
				self.state[done,:] = state_new
				self.cumm_reward[done] = 0.
				self.tracking_error[done] = 0.
		
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
		return self.tracking_error

	def _set_obs(self):
		self.obs[:,:self.observation_dims.shape[0]] = self.state[:,self.observation_dims]
		self.obs[:,self.observation_dims.shape[0]] = self.step_count
		reference_trajectory_start = th.as_tensor(self.step_count / self.horizon * self.th_reference_trajectory.shape[1], device=self.device, dtype=th.int)
		reference_trajectory_ids = th.unsqueeze(reference_trajectory_start, dim=1) + th.unsqueeze(th.arange(self.reference_trajectory_horizon, device=self.device, dtype=th.int), dim=0)
		reference_trajectory_ids = th.minimum(
			reference_trajectory_ids,
			th.as_tensor([self.th_reference_trajectory.shape[1]-1], device=self.device, dtype=th.int)
		)
		delta_reference = th.unsqueeze(self.state, dim=2) - th.transpose(self.th_reference_trajectory[:,reference_trajectory_ids], 0, 1)
		self.obs[:,(self.observation_dims.shape[0]+1):] = th.reshape(delta_reference, (self.num_envs, self.X_DIMS*self.reference_trajectory_horizon))

	def _interp_goal(self, step_count):
		ref_step_count = step_count / self.horizon * (self.th_reference_trajectory.shape[1]-1)
		lower_id = th.as_tensor(th.floor(ref_step_count), device=self.device, dtype=th.int)
		lower_id = th.minimum(lower_id, th.as_tensor([self.th_reference_trajectory.shape[1]-1], device=self.device, dtype=th.int))
		higher_id = th.minimum(lower_id+1, th.as_tensor([self.th_reference_trajectory.shape[1]-1], device=self.device, dtype=th.int))

		return th.transpose(self.th_reference_trajectory[:,lower_id] + (self.th_reference_trajectory[:,higher_id] - self.th_reference_trajectory[:,lower_id])*(ref_step_count - lower_id), 0, 1)

	def _update_tracking_error(self):
		goal_t = self._interp_goal(self.step_count)
		self.tracking_error += th.linalg.norm((self.state - goal_t)[:,self.th_tracking_dims], dim=1, keepdim=False)

	def _get_cost(self, action, state_):
		goal_t = self._interp_goal(self.step_count)
		y = (self.state - goal_t)[:,self.th_cost_dims]

		cost = th.sum((y @ self.th_Q) * y, dim=1, keepdim=False) * self.alpha_cost 
		cost = cost + th.sum(((action - self.th_u0) @ self.th_R) * (action - self.th_u0), dim=1, keepdim=False) * self.alpha_action_cost
		cost = cost * self.dt
		
		reached_goal = th.zeros((self.num_envs,), device=self.device, dtype=th.bool)

		return cost, reached_goal

	def _get_terminal_cost(self, done=None):
		if (done is not None):
			assert done.shape[0]==self.num_envs and type(done)==th.Tensor and done.dtype==th.bool, "done tensor of shape %d and type %s"%(done.shape[0], type(done))
		else:
			done = th.ones(self.num_envs, device=self.device, dtype=th.bool)

		goal_t = self._interp_goal(self.horizon * th.ones(self.num_envs, device=self.device, dtype=self.th_dtype))
		y = (self.state - goal_t)[:,self.th_cost_dims]

		cost = th.sum((y @ self.th_QT) * y, dim=1, keepdim=False) * self.alpha_terminal_cost
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

		# create the trajectory waypoints
		waypoint_size = [0.05, 0.05, 0.05]
		waypoint_reflectivity = 0.9
		waypoint_heading_color = 0xFFFFFF
		start_color = 0x008000
		interm_color = 0x884000
		end_color = 0xFF0000
		
		for ii in range(self.reference_trajectory.shape[1]):
			if (ii==0):
				waypoint_color = start_color
			elif (ii==(self.reference_trajectory.shape[1]-1)):
				waypoint_color = end_color
			else:
				waypoint_color = interm_color

			self.viz['traj_point_%d'%(ii)].set_object(meshcat.geometry.Ellipsoid(radii=waypoint_size),
			meshcat.geometry.MeshLambertMaterial(color=waypoint_color, reflectivity=waypoint_reflectivity))
			self.viz['traj_point_%d'%(ii)]['heading'].set_object(
				meshcat.geometry.TriangularMeshGeometry(
					vertices=np.array([[0., waypoint_size[1]/2, 0.], [0., -waypoint_size[1]/2, 0.], [waypoint_size[0]*2, 0., 0.]]),
					faces=np.array([[0, 1, 2]])
				),
				meshcat.geometry.MeshLambertMaterial(color=waypoint_heading_color, reflectivity=waypoint_reflectivity)
			)
			waypoint_pose = np.eye(4)
			waypoint_pose[:3,3] = self.reference_trajectory[:3,ii]
			waypoint_pose[:3,:3] = transfm.Rotation.from_euler('yxz', [self.reference_trajectory[4,ii], self.reference_trajectory[3,ii], self.reference_trajectory[5,ii]]).as_matrix()
			self.viz['traj_point_%d'%(ii)].set_transform(waypoint_pose)

	def render(self, mode='human'):
		if (not (hasattr(self, 'viz') and isinstance(self.viz, meshcat.Visualizer))):
			self._create_visualizer()
		pose = np.eye(4)
		pose[:3,3] = self.state[:3]
		pose[:3,:3] = transfm.Rotation.from_euler('yxz', [self.state[4], self.state[3], self.state[5]]).as_matrix()
		self.viz['root'].set_transform(pose)

	def close (self):
		pass