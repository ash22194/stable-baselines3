import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import meshcat
import scipy.spatial.transform as transfm
from scipy.io import loadmat
from scipy.interpolate import interp1d

class QuadcopterTT(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render_modes': ['human']}

	def __init__(
		self, trajectory_file, param=dict(), fixed_starts=False, reference_trajectory_horizon=0, normalized_actions=True, normalized_observations=True, alpha_cost=1., alpha_action_cost=1., alpha_terminal_cost=1.):
		# super(Quadcopter, self).__init__()
		# Define model paramters
		m = 0.5
		g = 9.81
		param_ = {'m': m, 'I': np.diag([4.86*1e-3, 4.86*1e-3, 8.8*1e-3]), 'l': 0.225, 'g': g, 'bk': 1.14*1e-7/(2.98*1e-6),
			# 'Q': np.diag([1, 1, 0.1, 0, 0, 0.1, 0.01, 0.01, 0.001, 0.001, 0.001, 0.0001]), 'R': 0.1*np.diag([0.002, 0.001, 0.001, 0.004]), 'QT': 2*np.eye(12),
			'Q': np.diag([1, 1, 1, 0, 0, 1, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]), 'R': 0.1*np.diag([0.002, 0.01, 0.01, 0.004]), 'QT': 2*np.eye(12),
			'u0': np.array([[m*g], [0.], [0.], [0.]]), 'T': 3, 'dt': 1e-3, 'lambda_': 1, 'X_DIMS': 12, 'U_DIMS': 4,
			'x_sample_limits': np.array([[-0.25, 0.25], [-0.25, 0.25], [-0.25, 0.25], [-np.pi/6, np.pi/6], [-np.pi/6, np.pi/6], [-np.pi/6, np.pi/6], [-1.25, 1.25], [-1.25, 1.25], [-1.25, 1.25], [-1.25, 1.25], [-1.25, 1.25], [-1.25, 1.25]]),
			'x_bounds': np.array([[-5., 5.], [-5., 5.], [-5, 5.], [-2*np.pi/3, 2*np.pi/3], [-2*np.pi/3, 2*np.pi/3], [-2*np.pi, 2*np.pi], [-6., 6.], [-6., 6.], [-6., 6.], [-6., 6.], [-6., 6.], [-6., 6.]]),
			'max_thrust_factor': 2
		}
		param_.update(param)
		param_['gamma_'] = np.exp(-param_['lambda_']*param_['dt'])
		param.update(param_)

		self.X_DIMS = param['X_DIMS'] # dimension of observations
		self.independent_sampling_dims = np.arange(self.X_DIMS)
		self.observation_dims = np.arange(self.X_DIMS)
		self.cost_dims = np.arange(self.X_DIMS)
		self.tracking_dims = np.array([0,1,2,3,4,5])
		self.U_DIMS = param['U_DIMS']

		self.u0 = param['u0']
		self.T = param['T']
		self.dt = param['dt']
		self.horizon = round(self.T / self.dt)

		assert trajectory_file.endswith('.mat'), 'Trajectory file must be a .mat'
		trajectory_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'examples/configs/quadcopter_tt/trajectories', trajectory_file)
		trajectory = loadmat(trajectory_file)['trajectory']
		trajectory_timestamps = np.linspace(0, self.horizon, trajectory.shape[1])
		self.goal = interp1d(x=trajectory_timestamps, y=trajectory)
		self.reference_trajectory_horizon = int(reference_trajectory_horizon/self.T*trajectory.shape[1])
		self.reference_trajectory = trajectory.copy()

		self.x_sample_limits = param['x_sample_limits'] # reset within these limits
		self.x_bounds = np.zeros(param['x_bounds'].shape)
		self.x_bounds[:,0] = param['x_bounds'][:,0] + np.min(trajectory, axis=1) 
		self.x_bounds[:,1] = param['x_bounds'][:,1] + np.max(trajectory, axis=1) 

		self.u_limits = np.zeros((self.U_DIMS, 2))
		max_thrust = param['max_thrust_factor']*m*g
		self.u_limits[0,1] = max_thrust
		self.u_limits[1,:] = np.array([-0.2*max_thrust, 0.2*max_thrust])
		self.u_limits[2,:] = np.array([-0.2*max_thrust, 0.2*max_thrust])
		self.u_limits[3,:] = np.array([-0.4*max_thrust, 0.4*max_thrust])
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
		self.fixed_starts = fixed_starts
		self.normalized_actions = normalized_actions
		self.normalized_observations = normalized_observations
		self.alpha_cost = alpha_cost
		self.alpha_action_cost = alpha_action_cost
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
		trajectory_input_obs_bounds_mid = np.tile(0.5*(self.x_bounds[self.observation_dims,1] + self.x_bounds[self.observation_dims,0]), self.reference_trajectory_horizon)
		trajectory_input_obs_bounds_range = np.tile(0.5*(self.x_bounds[self.observation_dims,1] - self.x_bounds[self.observation_dims,0]), self.reference_trajectory_horizon)

		self.obs_bounds_mid = np.concatenate((state_obs_bounds_mid, np.array([time_obs_bounds_mid]), trajectory_input_obs_bounds_mid))
		self.obs_bounds_range = np.concatenate((state_obs_bounds_range, np.array([time_obs_bounds_range]), trajectory_input_obs_bounds_range))			

		print('Observation dimension :', self.obs_bounds_mid.shape)
		print('X bounds :', self.x_bounds)

		if (normalized_observations):
			self.observation_space = spaces.Box(low=-1, high=1, shape=self.obs_bounds_mid.shape, dtype=np.float32)
			self.target_obs_mid = np.zeros(self.obs_bounds_mid.shape)
			self.target_obs_range = np.ones(self.obs_bounds_range.shape)

		else:
			self.observation_space = spaces.Box(
				low=self.obs_bounds_mid - self.obs_bounds_range, 
				high=self.obs_bounds_mid + self.obs_bounds_range, 
				dtype=np.float32
			)
			self.target_obs_mid = self.obs_bounds_mid
			self.target_obs_range = self.obs_bounds_range

	def step(self, action):
		# If scaling actions use this
		if (self.normalized_actions):
			action = 0.5*((self.u_limits[:,0] + self.u_limits[:,1]) + action*(self.u_limits[:,1] - self.u_limits[:,0]))

		state_ = self.dyn_rk4(self.state[:,np.newaxis], action[:,np.newaxis], self.dt)
		state_ = state_[:,0]
		state_ = np.minimum(self.x_bounds[:,1], np.maximum(self.x_bounds[:,0], state_))

		cost, reached_goal = self._get_cost(action, state_)
		reward = -cost

		self.state = state_
		self.step_count += 1
		self._update_tracking_error()

		if ((self.step_count >= self.horizon) or reached_goal):
			done = True
			terminal_cost = self._get_terminal_cost() # terminal cost
			reward -= terminal_cost
			info = {
				'ep_tracking_err': self.get_goal_dist(),
				'ep_terminal_cost': terminal_cost,
				'step_count' : self.step_count
			}
		else:
			done = False
			info = {'step_count' : self.step_count}

		return self.get_obs(), reward, done, False, info

	def reset(self, seed=None, options=None, state=None):
		super().reset(seed=seed)
		self.step_count = int(0*np.random.randint(low=0, high=self.horizon))
		if (self.fixed_starts):
			self.state = self.goal(self.step_count)
		else:
			if (state is None):
				self.state = (self.horizon - self.step_count) / self.horizon * (np.random.rand(self.independent_sampling_dims.shape[0]) - 0.5) * (self.x_sample_limits[:,1] - self.x_sample_limits[:,0])
				self.state += self.goal(self.step_count) + 0.5 * (self.x_sample_limits[:,0] + self.x_sample_limits[:,1])
			else:
				assert len(state.shape)==1 and state.shape[0]==self.X_DIMS, 'Invalid input state'
				self.state = state

		self.tracking_error = 0
		self._update_tracking_error()
		info = {'step_count' : self.step_count}

		return self.get_obs(), info

	def get_obs(self, normalized=None):
		obs = np.zeros(self.obs_bounds_mid.shape)
		obs[:self.observation_dims.shape[0]] = self.state[self.observation_dims]
		obs[self.observation_dims.shape[0]] = self.step_count
		reference_trajectory_start = int(self.step_count / self.horizon * self.reference_trajectory.shape[1])
		reference_trajectory_ids = np.minimum(
			self.reference_trajectory.shape[1]-1,
			np.arange(reference_trajectory_start, reference_trajectory_start+self.reference_trajectory_horizon)
		)
		delta_reference = self.state[:,np.newaxis] - self.reference_trajectory[:,reference_trajectory_ids]
		obs[(self.observation_dims.shape[0]+1):] = np.reshape(delta_reference, (self.reference_trajectory.shape[0]*self.reference_trajectory_horizon,), order='F')
		if ((normalized == None) and self.normalized_observations) or (normalized == True):
			obs = (obs - self.obs_bounds_mid) / self.obs_bounds_range
			obs = self.target_obs_range*obs + self.target_obs_mid
		return np.float32(obs)

	def get_goal_dist(self):
		return self.tracking_error
	
	def _update_tracking_error(self):
		x = self.state[:,np.newaxis]
		goal_t = self.goal(self.step_count)[:,np.newaxis]
		y = (x - goal_t)[self.tracking_dims,:]
		self.tracking_error += np.linalg.norm(y)

	def _get_cost(self, action, state_): #TODO
		x = self.state[:,np.newaxis]
		goal_t = self.goal(self.step_count)[:,np.newaxis]
		y = (x - goal_t)[self.cost_dims,:]

		a = action[:,np.newaxis]

		cost = np.sum(y * (self.Q @ y)) * self.alpha_cost + np.sum((a - self.u0) * (self.R @ (a - self.u0))) * self.alpha_action_cost
		cost = cost * self.dt

		reached_goal = False

		return cost, reached_goal

	def _get_terminal_cost(self): #TODO
		x = self.state[:,np.newaxis]
		goal_t = self.goal(self.horizon)[:,np.newaxis]
		y = (x - goal_t)[self.cost_dims,:]

		cost = np.sum(y * (self.QT @ y)) * self.alpha_terminal_cost

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

		x2 = x[3,0]
		x3 = x[4,0]
		x4 = x[5,0]
		x7 = x[8,0]
		x8 = x[9,0]
		x9 = x[10,0]
		x10 = x[11,0]

		u1 = u[0,0]
		u2 = u[1,0]
		u3 = u[2,0]
		u4 = u[3,0]

		t2 = np.cos(x2)
		t3 = np.cos(x3)
		t4 = np.cos(x4)
		t5 = np.sin(x2)
		t6 = np.sin(x3)
		t7 = np.sin(x4)
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
		t18 = np.sin(t11)
		t19 = np.sin(t12)
		t23 = 1.0/t3
		et1 = I1*t10*x9*x10-I3*t8*x9*x10+I1*I2*I3*x9*x10+I2*I3*l*t3*u2-I1*t6*t10*x8*x9+I3*t6*t8*x8*x9+I1*t9*t15*x9*x10-I2*t8*t15*x9*x10-I1*t10*t15*x9*x10+I3*t8*t15*x9*x10-I1*t10*t16*x9*x10+I3*t8*t16*x9*x10+I2*t10*t16*x9*x10-I3*t9*t16*x9*x10+I1*I2*I3*t6*x8*x9+I1*I2*bk*t2*t6*u4+I1*I3*l*t5*t6*u3+I1*t2*t3*t5*t9*t14-I2*t2*t3*t5*t8*t14-I1*t2*t3*t5*t10*t14+I2*t2*t3*t5*t10*t13+I3*t2*t3*t5*t8*t14-I3*t2*t3*t5*t9*t13-I1*t2*t5*t9*t14*t17+I2*t2*t5*t8*t14*t17+I1*t2*t5*t10*t14*t17-I3*t2*t5*t8*t14*t17
		et2 = -I2*t2*t5*t10*t14*t17+I3*t2*t5*t9*t14*t17-I1*t6*t9*t15*x8*x9+I2*t6*t8*t15*x8*x9+I1*t6*t10*t15*x8*x9-I3*t6*t8*t15*x8*x9-I1*t9*t15*t16*x9*x10+I2*t8*t15*t16*x9*x10+I1*t10*t15*t16*x9*x10-I3*t8*t15*t16*x9*x10-I2*t10*t15*t16*x9*x10*2.0+I3*t9*t15*t16*x9*x10*2.0-I1*t2*t3*t5*t6*t9*x8*x10+I2*t2*t3*t5*t6*t8*x8*x10+I1*t2*t3*t5*t6*t10*x8*x10-I3*t2*t3*t5*t6*t8*x8*x10

		dx = np.zeros((self.X_DIMS, 1))
		dx[0:6,:] = x[6:12,:]
		dx[6,0] = t22*u1*(t5*t7+t2*t4*t6)
		dx[7,0] = -t22*u1*(t4*t5-t2*t6*t7)
		dx[8,0] = -g+t2*t3*t22*u1
		dx[9,0] = (t20*t21*t23*(et1+et2))/I1
		dx[10,0] = t20*t21*(t9*t14*t19-t3*t9*x8*x10*2.0-t9*t18*x8*x9+t10*t18*x8*x9-I1*I2*t14*t19+I2*bk*t5*u4*2.0-I3*l*t2*u3*2.0+I1*I2*t3*x8*x10*2.0+I2*I3*t3*x8*x10*2.0+I1*I2*t18*x8*x9-I1*I3*t18*x8*x9-t3*t6*t9*t14*t15*2.0+t3*t6*t10*t14*t15*2.0+t3*t9*t15*x8*x10*2.0-t3*t10*t15*x8*x10*2.0+t2*t5*t6*t9*x9*x10*2.0-t2*t5*t6*t10*x9*x10*2.0+I1*I2*t3*t6*t14*t15*2.0-I1*I3*t3*t6*t14*t15*2.0-I1*I2*t3*t15*x8*x10*2.0+I1*I3*t3*t15*x8*x10*2.0-I1*I2*t2*t5*t6*x9*x10*2.0+I1*I3*t2*t5*t6*x9*x10*2.0)*(-1.0/2.0)
		dx[11,0] = t20*t21*t23*(-t10*x8*x9+t6*t10*x9*x10-t9*t15*x8*x9+t10*t15*x8*x9+I1*I3*x8*x9+I2*I3*x8*x9+I2*bk*t2*u4+I3*l*t5*u3-I1*I3*t6*x9*x10+I2*I3*t6*x9*x10+I1*I2*t15*x8*x9-I1*I3*t15*x8*x9+t6*t9*t15*x9*x10-t6*t10*t15*x9*x10+t2*t3*t5*t6*t9*t14-t2*t3*t5*t6*t10*t14-t2*t3*t5*t9*x8*x10+t2*t3*t5*t10*x8*x10-I1*I2*t6*t15*x9*x10+I1*I3*t6*t15*x9*x10-I1*I2*t2*t3*t5*t6*t14+I1*I3*t2*t3*t5*t6*t14+I1*I2*t2*t3*t5*x8*x10-I1*I3*t2*t3*t5*x8*x10)

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
		mid_point = int(self.reference_trajectory.shape[1] / 2.)
		self.viz['root'].set_cam_pos([self.reference_trajectory[0, mid_point], self.reference_trajectory[1, mid_point]-3., self.reference_trajectory[2, mid_point]+2.])
		self.viz['root'].set_cam_target([self.state[0], self.state[1], self.state[2]])
		return self.viz['root'].get_image()

	def close (self):
		pass