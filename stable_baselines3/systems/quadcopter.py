import gymnasium as gym
from gymnasium import spaces
import numpy as np
from copy import deepcopy
import meshcat
import scipy.spatial.transform as transfm

class Quadcopter(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render_modes': ['human']}

	def __init__(self, sys=dict(), fixed_start=False, normalized_actions=True, normalized_observations=False):
		# super(Quadcopter, self).__init__()
		# Define model paramters
		m = 0.5
		g = 9.81
		sys_ = {'m': m, 'I': np.diag([4.86*1e-3, 4.86*1e-3, 8.8*1e-3]), 'l': 0.225, 'g': g, 'bk': 1.14*1e-7/(2.98*1e-6),\
				'Q': np.diag([5, 0.001, 0.001, 5, 0.5, 0.5, 0.05, 0.075, 0.075, 0.05]), 'R': np.diag([0.002, 0.01, 0.01, 0.004]),\
				'goal': np.array([[1.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]), 'u0': np.array([[m*g], [0.], [0.], [0.]]),\
				'T': 4, 'dt': 1e-3, 'lambda_': 1, 'X_DIMS': 12, 'U_DIMS': 4,\
				'x_sample_limits': np.array([[-2., 2.], [-2., 2.], [0.6, 1.4], [-np.pi/5, np.pi/5], [-np.pi/5, np.pi/5], [-2*np.pi/5, 2*np.pi/5], [-5., 5.], [-5., 5.], [-5., 5.], [-3., 3.], [-3., 3.], [-3., 3.]]),\
				'x_bounds': np.array([[-10., 10.], [-10., 10.], [0., 2.], [-2*np.pi/3, 2*np.pi/3], [-2*np.pi/3, 2*np.pi/3], [-2*np.pi, 2*np.pi], [-10., 10.], [-10., 10.], [-10., 10.], [-7., 7.], [-7., 7.], [-7., 7.]]),\
				'u_limits': np.array([[0, 2*m*g], [-0.25*m*g, 0.25*m*g], [-0.25*m*g, 0.25*m*g], [-0.125*m*g, 0.125*m*g]])}
		sys_.update(sys)
		sys_['gamma_'] = np.exp(-sys_['lambda_']*sys_['dt'])
		sys.update(sys_)

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
		self.R = sys['R']
		self.gamma_ = sys['gamma_']
		self.lambda_ = sys['lambda_']

		self.m = sys['m']
		self.l = sys['l']
		self.g = sys['g']
		self.bk = sys['bk']
		self.I = sys['I']
		self.fixed_start = fixed_start
		self.normalized_actions = normalized_actions
		self.normalized_observations = normalized_observations
		# self.reset()

		# Define action and observation space
		# They must be gym.spaces objects
		# Example when using continuous actions:
		if (normalized_actions):
			self.action_space = spaces.Box(low=-1, high=1, shape=(self.U_DIMS,), dtype=np.float32)
		else:
			self.action_space = spaces.Box(low=self.u_limits[:,0], high=self.u_limits[:,1], dtype=np.float32)

		if (normalized_observations):
			self.observation_space = spaces.Box(low=-1, high=1, shape=(self.observation_dims.shape[0],), dtype=np.float32)
		else:
			self.observation_space = spaces.Box(low=self.x_bounds[self.observation_dims,0], high=self.x_bounds[self.observation_dims,1], dtype=np.float32)

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

		if ((self.step_count >= self.horizon) or reached_goal):
			done = True
			info = {'terminal_state': deepcopy(self.state), 'step_count' : deepcopy(self.step_count)}
		else:
			done = False
			info = {'terminal_state': np.array([]), 'step_count' : deepcopy(self.step_count)}

		return self.get_obs(), reward, done, False, info

	def reset(self, seed=None, options=None, state=None):
		super().reset(seed=seed)
		if (state is None):
			if (self.fixed_start):
				self.state = np.array([0., 0., 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
			else:
				self.state = 0.5 * (self.x_sample_limits[:,0] + self.x_sample_limits[:,1]) + (np.random.rand(self.independent_sampling_dims.shape[0]) - 0.5) * (self.x_sample_limits[:,1] - self.x_sample_limits[:,0])
		else:
			assert len(state.shape)==1 and state.shape[0]==self.X_DIMS, 'Invalid input state'
			self.state = state

		self.step_count = 0
		info = {'terminal_state': np.array([]), 'step_count' : deepcopy(self.step_count)}

		return self.get_obs(), info

	def get_obs(self, normalized=None):
		obs = self.state[self.observation_dims]
		if ((normalized == None) and self.normalized_observations) or (normalized == True):
			obs_bounds_mid = 0.5*(self.x_bounds[self.observation_dims,1] + self.x_bounds[self.observation_dims,0])
			obs_bounds_range = 0.5*(self.x_bounds[self.observation_dims,1] - self.x_bounds[self.observation_dims,0])
			obs = (obs - obs_bounds_mid) / obs_bounds_range

		return np.float32(obs)

	def _get_cost(self, action, state_):
		x = self.get_obs(normalized=False)
		x = x[:,np.newaxis]
		a = action[:,np.newaxis]

		cost = np.sum((x - self.goal) * (self.Q @ (x - self.goal))) + np.sum((a - self.u0) * (self.R @ (a - self.u0)))
		cost = cost * self.dt

		reached_goal = np.linalg.norm(state_[self.observation_dims] - self.goal[:,0]) <= 1e-2	

		return cost, reached_goal

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
		t21 = 1.0/I2
		t22 = 1.0/I3
		t23 = 1.0/m
		t15 = t2**2
		t16 = t3**2
		t17 = t3**3
		t18 = np.sin(t11)
		t19 = np.sin(t12)
		t20 = t5**2
		t24 = 1.0/t3
		t25 = 1.0/t16
		et1 = I1*t10*x9*x10-I3*t8*x9*x10+I1*I2*I3*x9*x10+I2*I3*l*t3*u2-I1*t6*t10*x8*x9+I3*t6*t8*x8*x9+I1*t9*t15*x9*x10-I2*t8*t15*x9*x10-I1*t10*t15*x9*x10+I3*t8*t15*x9*x10-I1*t10*t16*x9*x10+I3*t8*t16*x9*x10+I2*t10*t16*x9*x10-I3*t9*t16*x9*x10+I1*I2*I3*t16*x8*x9+I1*I2*bk*t2*t6*u4+I1*I3*l*t5*t6*u3+I1*t2*t3*t5*t9*t14-I2*t2*t3*t5*t8*t14-I1*t2*t3*t5*t10*t14+I2*t2*t3*t5*t10*t13+I3*t2*t3*t5*t8*t14-I3*t2*t3*t5*t9*t13-I1*t2*t5*t9*t14*t17+I2*t2*t5*t8*t14*t17+I1*t2*t5*t10*t14*t17-I3*t2*t5*t8*t14*t17
		et2 = -I2*t2*t5*t10*t14*t17+I3*t2*t5*t9*t14*t17-I1*t6*t9*t15*x8*x9+I2*t6*t8*t15*x8*x9+I1*t6*t10*t15*x8*x9-I3*t6*t8*t15*x8*x9-I1*t9*t15*t16*x9*x10+I2*t8*t15*t16*x9*x10+I1*t10*t15*t16*x9*x10-I3*t8*t15*t16*x9*x10-I2*t10*t15*t16*x9*x10*2.0+I3*t9*t15*t16*x9*x10*2.0+I1*I2*I3*t6*t15*x8*x9-I1*I2*I3*t15*t16*x8*x9-I1*I2*I3*t2*t5*t17*x8*x10-I1*t2*t3*t5*t6*t9*x8*x10+I2*t2*t3*t5*t6*t8*x8*x10+I1*t2*t3*t5*t6*t10*x8*x10-I3*t2*t3*t5*t6*t8*x8*x10+I1*I2*I3*t2*t3*t5*t6*x8*x10
		mt2 = t21*t22*(t9*t14*t19-t3*t9*x8*x10*2.0-t9*t18*x8*x9+t10*t18*x8*x9-I1*I2*t14*t19+I2*bk*t5*u4*2.0-I3*l*t2*u3*2.0+I1*I2*t3*x8*x10*2.0+I2*I3*t3*x8*x10*2.0+I1*I2*t18*x8*x9-I1*I3*t18*x8*x9-t3*t6*t9*t14*t15*2.0+t3*t6*t10*t14*t15*2.0+t3*t9*t15*x8*x10*2.0-t3*t10*t15*x8*x10*2.0+t2*t5*t6*t9*x9*x10*2.0-t2*t5*t6*t10*x9*x10*2.0+I1*I2*t3*t6*t14*t15*2.0-I1*I3*t3*t6*t14*t15*2.0-I1*I2*t3*t15*x8*x10*2.0+I1*I3*t3*t15*x8*x10*2.0-I1*I2*t2*t5*t6*x9*x10*2.0+I1*I3*t2*t5*t6*x9*x10*2.0)*(-1.0/2.0)
		mt3 = t25*(t5*x9-t2*t3*x10)*(t3*t5*x8-t2*t6*x9)+t21*t22*t24*(-t9*t15*x8*x9-t10*t20*x8*x9+I2*bk*t2*u4+I3*l*t5*u3+I1*I2*t15*x8*x9+I1*I3*t20*x8*x9+t6*t9*t15*x9*x10+t6*t10*t20*x9*x10+t2*t3*t5*t6*t9*t14-t2*t3*t5*t6*t10*t14-t2*t3*t5*t9*x8*x10+t2*t3*t5*t10*x8*x10-I1*I2*t6*t15*x9*x10-I1*I3*t6*t20*x9*x10-I1*I2*t2*t3*t5*t6*t14+I1*I3*t2*t3*t5*t6*t14+I1*I2*t2*t3*t5*x8*x10-I1*I3*t2*t3*t5*x8*x10)+t25*x8*np.cos(x2-x3)*(t2*x9+t3*t5*x10)

		dx = np.zeros((self.X_DIMS, 1))
		dx[0:6,:] = x[6:12,:]
		dx[6,0] = t23*u1*(t5*t7+t2*t4*t6)
		dx[7,0] = -t23*u1*(t4*t5-t2*t6*t7)
		dx[8,0] = -g+t2*t3*t23*u1
		dx[9,0] = (t21*t22*t24*(et1+et2))/I1
		dx[10,0] = mt2
		dx[11,0] = mt3

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

	def render(self, mode='human'):
		if (not (hasattr(self, 'viz') and isinstance(self.viz, meshcat.Visualizer))):
			self._create_visualizer()
		pose = np.eye(4)
		pose[:3,3] = self.state[:3]
		pose[:3,:3] = transfm.Rotation.from_euler('yxz', [self.state[4], self.state[3], self.state[5]]).as_matrix()
		self.viz['root'].set_transform(pose)

	def close (self):
		pass