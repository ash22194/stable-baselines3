import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import meshcat
import scipy.spatial.transform as transfm
from scipy.io import loadmat
from scipy.interpolate import interp1d


class UnicycleTT(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render_modes': ['human']}

	def __init__(
		self, trajectory_file, param=dict(), fixed_starts=False, intermittent_starts=False, reference_trajectory_horizon=0, normalized_actions=True, normalized_observations=True, nonlinear_cost=True, alpha_track_cost=1., alpha_balance_cost=1., alpha_action_cost=1., alpha_terminal_cost=1.):

		# Define model parameters
		mw = 0.5
		mf = 0.65 + mw
		md = 2.64 * 2.

		rw = 0.1524
		frame_length = 0.39
		rf = frame_length / 2.
		rd = frame_length / 2.
		upper_body_length = 0.4

		param_ = {'mw': mw, 'mf': mf, 'md': md, 'rw': rw, 'rf': rf, 'rd': rd, \
			'Iw': np.diag([mw*(rw**2 + 0.04**2)/5, 2*mw*(rw**2)/5, mw*(rw**2 + 0.04**2)/5]),\
			'If': np.diag([mf*(frame_length**2 + 0.08**2)/12, mf*(frame_length**2 + 0.08**2)/12, 2*mf*(0.08**2)/12]),\
			'Id': np.diag([md*(upper_body_length**2 + 0.08**2)/12, md*(upper_body_length**2 + 0.08**2)/12, 2*md*(0.08**2)/12]),\
			'alpha': -np.pi/2, 'g': 9.81, 'fcoeff': 0.05, \
			'u0': np.zeros((2,1)), 'T': 6, 'dt':7.5e-3, 'gamma_':0.9999, 'X_DIMS': 16, 'U_DIMS': 2, \
			'Q_nonlinear_track': np.diag([1,1,0,1,0.001,0.001,0,0.001]), 'Q_nonlinear_balance': np.diag([0.01,0.001, 0.00001,0.000001, 0.0001,0.000001]),\
			'R_nonlinear': (np.eye(2) / 5000.), 'QT_nonlinear': 2*np.eye(14)/1000, \
			'x_sample_limits': np.array([[-0.1, 0.1], [-0.1, 0.1], [-np.pi/12, np.pi/12], [-np.pi/12, np.pi/12], [-np.pi/12, np.pi/12], [-np.pi, np.pi], [-np.pi/6, np.pi/6], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [5., 15.], [-0.5, 0.5]]),\
			'x_bounds': np.array([[-20., 20.], [-20., 20.], [0., 2.], [-2*np.pi, 2*np.pi], [-np.pi/2.5, np.pi/2.5], [-np.pi/2.5, np.pi/2.5], [-10*np.pi, 10*np.pi], [-4*np.pi/3, 4*np.pi/3], [-8, 8], [-8, 8], [-8, 8], [-8., 8.], [-8., 8.], [-8., 8.], [-5., 25.], [-8., 8.]]),\
			'u_limits': np.array([[-15., 15.], [-15., 15.]])
		}
		param_.update(param)
		param.update(param_)

		self.X_DIMS = param['X_DIMS'] # dimension of observations
		self.independent_dims = np.array([0,1,3,4,5,6,7,11,12,13,14,15]) # the same length as x_sample_limits
		self.observation_dims = np.array([0,1,3,4,5,7,8,9,11,12,13,14,15])
		self.cost_dims = np.array([0,1,3,4,5,7,8,9,11,12,13,14,15])
		self.tracking_dims = np.array([0,1,3,4,5,6,7])
		self.U_DIMS = param['U_DIMS']

		self.u0 = param['u0']
		self.T = param['T']
		self.dt = param['dt']
		self.horizon = round(self.T / self.dt)

		assert trajectory_file.endswith('.mat'), 'Trajectory file must be a .mat'
		trajectory_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'examples/configs/unicycle_tt/trajectories', trajectory_file)
		trajectory = loadmat(trajectory_file)['trajectory']
		self.reference_trajectory_horizon = int(reference_trajectory_horizon/self.T*trajectory.shape[1])
		self.reference_trajectory = trajectory.copy()
		self.goal = self._interp_goal(np.arange(self.horizon))

		self.x_sample_limits = param['x_sample_limits'] # reset within these limits
		self.x_bounds = np.zeros(param['x_bounds'].shape)
		self.x_bounds[:,0] = param['x_bounds'][:,0] + np.min(trajectory, axis=1) 
		self.x_bounds[:,1] = param['x_bounds'][:,1] + np.max(trajectory, axis=1)

		self.u_limits = param['u_limits']
		self.nonlinear_cost = nonlinear_cost
		if (nonlinear_cost):
			self.QT = param['QT_nonlinear']*alpha_terminal_cost
			self.Q = np.zeros(self.QT.shape)
			self.Q[:param['Q_nonlinear_track'].shape[0],:param['Q_nonlinear_track'].shape[1]] = param['Q_nonlinear_track']*alpha_track_cost
			self.Q[param['Q_nonlinear_track'].shape[0]:,param['Q_nonlinear_track'].shape[1]:] = param['Q_nonlinear_balance']*alpha_balance_cost
			self.R = param['R_nonlinear']*alpha_action_cost
		else:
			NotImplementedError

		self.md = param['md']
		self.mf = param['mf']
		self.mw = param['mw']
		self.rd = param['rd']
		self.rf = param['rf']
		self.rw = param['rw']
		self.Id = param['Id']
		self.If = param['If']
		self.Iw = param['Iw']

		self.g = param['g']
		self.alpha = param['alpha']
		self.fcoeff = param['fcoeff']
		self.fixed_starts = fixed_starts
		self.normalized_actions = normalized_actions
		self.normalized_observations = normalized_observations
		if (intermittent_starts):
			self.step_count_high = self.horizon
		else:
			self.step_count_high = 1

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
		state_ = self._enforce_bounds(state_[:,0])
		
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
		self.step_count = int(np.random.randint(low=0, high=self.step_count_high))
		if (self.fixed_starts):
			sample = self.goal[self.independent_dims,self.step_count]
		else:
			if (state is None):
				sample = (self.horizon - self.step_count) / self.horizon * (np.random.rand(self.independent_dims.shape[0]) - 0.5) * (self.x_sample_limits[:,1] - self.x_sample_limits[:,0])
				sample += self.goal[self.independent_dims,self.step_count] + 0.5 * (self.x_sample_limits[:,0] + self.x_sample_limits[:,1])
			else:
				assert len(state.shape)==1 and state.shape[0]==self.X_DIMS, 'Invalid input state'
				# construct the dependent states from the independent
				sample = state[self.independent_dims]

		self.state = np.zeros(self.X_DIMS)
		self.state[self.independent_dims] = sample
		self.state = self._enforce_bounds(self.state)

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
		obs[(self.observation_dims.shape[0]+1):] = np.reshape(delta_reference[self.observation_dims,:], (self.observation_dims.shape[0]*self.reference_trajectory_horizon,), order='F')
		if ((normalized == None) and self.normalized_observations) or (normalized == True):
			obs = (obs - self.obs_bounds_mid) / self.obs_bounds_range
			obs = self.target_obs_range*obs + self.target_obs_mid
		return np.float32(obs)
	
	def get_goal_dist(self):
		return self.tracking_error
	
	def _interp_goal(self, step_count):
		ref_step_count = step_count / self.horizon * (self.reference_trajectory.shape[1] - 1)
		lower_id = np.minimum(np.floor(ref_step_count), self.reference_trajectory.shape[1] - 1).astype(np.int32)
		higher_id = np.minimum(lower_id+1, self.reference_trajectory.shape[1] - 1).astype(np.int32)

		return (self.reference_trajectory[:,lower_id] + (self.reference_trajectory[:,higher_id] - self.reference_trajectory[:,lower_id])*(ref_step_count - lower_id))

	def _update_tracking_error(self):
		x = self.state[:,np.newaxis]
		t = min(self.step_count, self.horizon-1)
		goal_t = self.goal[:,t:(t+1)]
		y = (x - goal_t)[self.tracking_dims,:]

		self.tracking_error += np.linalg.norm(y)

	def _get_cost(self, action, state_):
		if (self.nonlinear_cost):
			y = self.get_taskspace_obs()
		else:
			t = min(self.step_count, self.horizon-1)
			y = (self.state - self.goal[:,t])[self.cost_dims]

		y = y[:,np.newaxis]
		a = action[:,np.newaxis]

		cost = y.T @ (self.Q @ y) + (a - self.u0).T @ (self.R @ (a - self.u0))
		cost = cost[0,0] * self.dt

		reached_goal = False

		return cost, reached_goal
	
	def _get_terminal_cost(self):
		if (self.nonlinear_cost):
			y = self.get_taskspace_obs()
		else:
			goal_t = self.goal[:,(self.horizon-1)]
			y = (self.state - goal_t)[self.cost_dims]
		
		y = y[:,np.newaxis]

		cost = y.T @ (self.QT @ y)

		return cost[0,0]
	
	def get_taskspace_obs(self):
		x = self.state
		t = min(self.step_count, self.horizon-1)
		goal_t = self.goal[:,t:(t+1)]

		# # calculate com position and velocity
		# Rframe = transfm.Rotation.from_euler('yxz', [x[5], x[4], x[3]]).as_matrix()
		# p_wheel = Rframe @ np.array([0.,0.,-self.rf])
		# p_dumbell = Rframe @ np.array([0.,0.,self.rd])

		# p_com = (self.mw*p_wheel + self.md*p_dumbell) / (self.mf + self.mw + self.md)
		# v_com = self._jacobian_com(x) @ x[8:16]

		# Rwheel = transfm.Rotation.from_euler('yxz', [x[5]+x[6], x[4], x[3]]).as_matrix()
		# p_con = p_wheel + Rwheel @ np.array([self.rw*np.sin(x[5]+x[6]), 0., -self.rw*np.cos(x[5]+x[6])])
		# v_con = self._jacobian_contact_trace(x) @ x[8:16]

		# Ryaw = transfm.Rotation.from_euler('yxz', [0., 0., -x[3]]).as_matrix()
		# delta_com = Ryaw @ (p_com - p_con)
		# vdelta_com = Ryaw @ (v_com - v_con)
		# delta_com = (p_com - p_con)
		# vdelta_com = (v_com - v_con)

		ph = x[3]
		th = x[4]
		om = x[5]

		v_ph = x[11]
		v_th = x[12]
		v_om = x[13]

		t2 = np.cos(om)
		t3 = np.cos(ph)
		t4 = np.cos(th)
		t5 = np.sin(om)
		t6 = np.sin(ph)
		t7 = np.sin(th)

		t10 = self.rf*t2
		t14 = self.rf*t3*t5
		t15 = 1.0/(self.md+self.mf+self.mw)
		t13 = self.rw+t10
		t16 = self.md*self.rd-self.mw*self.rf

		y = np.zeros(14)
		y[:4] = x[:4]    # x-y-z-yaw
		y[4:8] = x[8:12] # vx-vy-vz-vyaw
		# y[8:10] = delta_com[:2]
		# y[10:12] = delta_vcom[:2]
		y[8] = t14+t15*t16*(t3*t5+t2*t6*t7)+t6*t7*t13
		y[9] = t15*t16*(t5*t6-t2*t3*t7)+self.rf*t5*t6-t3*t7*t13
		# delta_com_contact_z = t4*t13+t2*t4*t15*t16

		y[10] = t3*t10*(v_om+t7*v_ph)+t15*t16*(t2*t3*v_om-t5*t6*v_ph-t5*t6*t7*v_om+t2*t3*t7*v_ph+t2*t4*t6*v_th)+self.rw*t3*t7*v_ph+t4*t6*t13*v_th-self.rf*t5*t6*(v_ph+t7*v_om)
		y[11] = v_ph*(t14+self.rw*t6*t7+t6*t7*t10)+self.rf*v_om*(t2*t6+t3*t5*t7)+t15*t16*(t2*t6*v_om+t3*t5*v_ph+t3*t5*t7*v_om+t2*t6*t7*v_ph-t2*t3*t4*v_th)-t3*t4*t13*v_th
		# delta_vcom_contact_z = -t15*t16*(t4*t5*v_om+t2*t7*v_th)-t7*t13*v_th-self.rf*t4*t5*v_om

		y[12] = x[7] - goal_t[7,0]
		y[13] = x[15] - goal_t[15,0]

		return y

	def _enforce_bounds(self, state_):
		state_[self.independent_dims] = np.maximum(self.x_bounds[self.independent_dims,0], np.minimum(self.x_bounds[self.independent_dims,1], state_[self.independent_dims]))
		err_p, err_v = self._err_posvel_contact(state_)
		state_[:3] += (-err_p)
		state_[8:11] += (-err_v)

		return state_

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

		md = self.md
		mf = self.mf
		mw = self.mw
		rd = self.rd
		rf = self.rf
		rw = self.rw
		Idxx = self.Id[0,0]
		Idyy = self.Id[1,1]
		Idzz = self.Id[2,2]
		Ifxx = self.If[0,0]
		Ifyy = self.If[1,1]
		Ifzz = self.If[2,2]
		Iwxx = self.Iw[0,0]
		Iwyy = self.Iw[1,1]
		Iwzz = self.Iw[2,2]

		g = self.g
		alpha = self.alpha
		fcoeff = self.fcoeff

		ph = x[3,0]
		th = x[4,0]
		om = x[5,0]
		ps = x[6,0] + om
		ne = x[7,0]

		vx = x[8,0]
		vy = x[9,0]
		vz = x[10,0]
		v_ph = x[11,0]
		v_th = x[12,0]
		v_om = x[13,0]
		v_ps = x[14,0] + v_om
		v_ne = x[15,0]

		Tomega = -u[0,0]
		Tneta = u[1,0]

		t2 = np.cos(ne)
		t3 = np.cos(om)
		t4 = np.cos(th)
		t5 = np.sin(ne)
		t6 = np.cos(ps)
		t7 = np.cos(ph)
		t8 = np.sin(om)
		t9 = np.sin(th)
		t10 = np.sin(ps)
		t11 = np.sin(ph)
		t12 = alpha+om
		t13 = md+mf
		t14 = md*rd
		t15 = mf*rf
		t16 = mw*rf
		t17 = rd+rf
		t18 = Idxx*2.0
		t19 = Idyy*2.0
		t20 = Idyy*4.0
		t21 = Idzz*2.0
		t22 = Idzz*4.0
		t23 = Ifxx*2.0
		t24 = Ifyy*2.0
		t25 = Ifyy*4.0
		t26 = Ifzz*2.0
		t27 = Iwxx*2.0
		t28 = Iwyy*2.0
		t29 = Iwyy*4.0
		t30 = Iwzz*2.0

		t32 = ne*2.0
		t33 = om*2.0
		t34 = th*2.0
		t35 = v_om**2
		t36 = v_th**2
		t37 = v_ph**2
		t38 = ps*2.0
		t57 = v_om*v_ph*2.0
		t63 = -Idyy
		t64 = -Idzz
		t70 = -Iwzz
		t39 = np.cos(t32)
		t40 = t2**2
		t41 = np.cos(t33)
		t42 = np.cos(t34)
		t43 = t4**2
		t44 = np.sin(t32)
		t45 = t5**2
		t46 = np.cos(t38)
		t47 = t6**2
		t48 = t14*2.0
		t49 = t14*4.0
		t50 = t16*2.0
		t51 = t16*4.0
		t52 = np.sin(t33)
		t53 = t8**2
		t54 = np.sin(t34)
		t55 = t9**2
		t56 = np.sin(t38)
		t58 = np.cos(t12)
		t59 = np.sin(t12)
		t60 = rf*t3
		t61 = mw+t13
		t62 = -t18
		t65 = -t21
		t66 = -t22
		t67 = -t23
		t68 = -t26
		t69 = -t27
		t71 = -t30
		t72 = t9*v_ph
		t73 = rf*t13
		t74 = md*t17
		t75 = t3*t11
		t76 = -t16
		t80 = t9*t57
		t82 = Idyy+t64
		t83 = Iwxx+t70
		t86 = t14*t17
		t91 = t7*t8*t9
		t92 = t35+t37
		t95 = t12*2.0
		t77 = -t50
		t78 = -t51
		t79 = t59**2
		t81 = Idxx*t58
		t84 = rw+t60
		t85 = rw*t61
		t87 = t72+v_ps
		t88 = Idzz*t40
		t89 = Idyy*t45
		t90 = t58**2
		t93 = t17*t48
		t94 = Idyy+Idzz+t62
		t96 = Idxx*t4*t59
		t97 = np.cos(t95)
		t98 = np.sin(t95)
		t99 = t19+t65
		t100 = t20+t66
		t101 = t27+t71
		t102 = t14+t73
		t103 = t15+t74
		t104 = t14+t76
		t106 = t9*t92
		t110 = t39*t82
		t114 = t80+t92
		t119 = t75+t91
		t128 = (t44*t59*t82)/2.0
		t132 = (t4*t44*t58*t82)/2.0
		t105 = -t96
		t107 = t97+3.0
		t108 = t48+t77
		t109 = t49+t78
		t111 = t110*2.0
		t112 = t3*t102
		t113 = t3*t103
		t115 = t46*t101
		t116 = t56*t101
		t117 = Idxx+t110
		t118 = rw*t3*t104
		t123 = t23+t68+t93
		t124 = t57+t106
		t125 = t94*t97
		t129 = t39*t90*t99
		t135 = t42*t44*t58*t100
		t136 = t4*t44*t82*t98

		t140 = -t98*(t94-t110)
		t120 = -t111
		t121 = rw*t3*t108
		t122 = rw*t3*t109
		t127 = -t125
		t130 = t85+t112
		t131 = t85+t113
		t133 = t41*t123
		t134 = t52*t123

		t126 = t18+t120
		t143 = t134+t140

		M = np.zeros((8,8))

		M[0,0] = -t8*t11*t103+t7*t9*t130
		M[1,0] = t4*t11*t131
		M[2,0] = t103*(t3*t7-t8*t9*t11)
		M[3,0] = t7*t85
		M[5,0] = 1.0

		M[0,1] = t7*t8*t103+t9*t11*t130
		M[1,1] = -t4*t7*t131
		M[2,1] = t103*t119
		M[3,1] = t11*t85
		M[6,1] = 1.0

		M[1,2] = -t9*t131
		M[2,2] = -t4*t8*t103
		M[7,2] = 1.0

		M[0,3] = t53*t86+(t55*(Idyy+Idzz+t24+t28+t86+t110+t121+t41*t86))/2.0+t43*(t90*(t88+t89)+Idxx*t79+Ifxx*t53+Iwzz*t47+Ifzz*t3**2+Iwxx*t10**2)+(t44*t54*t58*t82)/2.0
		M[1,3] = t4*(t116+t143+rw*t8*t109)*(-1.0/4.0)+t2*t5*t9*t59*t82
		M[2,3] = t132+(t9*(Idyy+Idzz+t24+t93+t110))/2.0
		M[3,3] = t9*(Iwyy+t118)
		M[4,3] = t105
		M[5,3] = rf*t8*t11-t7*t9*t84
		M[6,3] = -rf*t7*t8-t9*t11*t84

		M[0,4] = t9*t128-(t4*(t56*t83+t52*(Ifxx-Ifzz+t86)-t98*(-Idxx+t88+t89)))/2.0
		M[1,4] = Idxx/2.0+Idyy/4.0+Idzz/4.0+Ifxx/2.0+Ifzz/2.0+Iwxx/2.0+Iwzz/2.0+t86/2.0+t115/4.0+t122/4.0-t125/4.0+t133/4.0-(t39*t79*t99)/4.0
		M[2,4] = t128
		M[3,4] = 0.0
		M[4,4] = t81
		M[5,4] = -t4*t11*t84
		M[6,4] = t4*t7*t84
		M[7,4] = t9*t84

		M[0,5] = t132+(t9*(Idyy+Idzz+t24+t93+t110+t121))/2.0
		M[1,5] = t128
		M[2,5] = Idyy/2.0+Idzz/2.0+Ifyy+t86+t110/2.0
		M[3,5] = t118
		M[5,5] = -t7*t60+rf*t8*t9*t11
		M[6,5] = -rf*t119
		M[7,5] = rf*t4*t8

		M[0,6] = Iwyy*t9
		M[3,6] = Iwyy
		M[5,6] = -rw*t7
		M[6,6] = -rw*t11

		M[0,7] = t105
		M[1,7] = t81
		M[4,7] = Idxx

		t139 = t9*t59*t126
		et1 = t35*(rw*t8*t9*t104*4.0+t2*t4*t5*t59*t82*4.0)*(-1.0/4.0)-(v_ne*(Idxx*(t4*t58*v_om-t9*t59*v_th)*8.0-t39*t100*(t9*t59*v_th*2.0+t4*t58*(t72*2.0+v_om)*2.0)+t44*t82*(t9*v_om*8.0+v_ph*(t79*4.0-t42*t107*2.0)-t4*t98*v_th*4.0)))/8.0+(t36*(t9*(t116+t143)+t4*t44*t59*t99))/4.0+(v_th*(v_ph*(t135+t54*(t25+t29+t67+t68+t69+t71+t93+t94+t115+t122+t127+t133+t107*t110))+t4*(v_om*(Idyy+Idzz+t24+t93+t125-t133+t39*t79*t99)+v_ps*(t28-t46*t83*2.0))*2.0))/4.0+v_ph*(fcoeff-rw*t8*t72*t104+t43*t56*t83*v_ps)
		et2 = v_om*v_ph*(t43*t143*-2.0+rw*t8*t55*t109*2.0+t44*t54*t59*t82*2.0)*(-1.0/4.0)

		nle = np.zeros((8,1))
		nle[0,0] = et1+et2
		nle[1,0] = t37*(t135+t54*(t21+t25+t29+t62+t63+t67+t68+t69+t71+t93+t115+t122+t133+t18*t97-t89*t97*2.0+t40*(Idyy+t64*t90)*4.0+t39*(Idyy+t65)))*(-1.0/8.0)-(v_ph*(v_om*(t4*(Idyy+Idzz+t24+t93+t122+t127+t129+t133)-t9*t44*t58*t99)+t4*v_ps*(Iwyy+t46*t83)*2.0))/2.0-(v_th*(v_om*(t134*2.0-t98*(t94-t110)*2.0+rw*t8*t109*2.0)+t56*v_ps*(Iwxx*4.0-Iwzz*4.0)))/4.0+(v_ne*(v_ph*(t136-t139)-t59*t126*v_om+t44*t79*t99*v_th))/2.0-g*t9*t131+(t35*t44*t58*t100)/8.0
		nle[2,0] = -Tomega+(t36*t143)/4.0-v_ne*(v_ph*(t9*t44*t82-t4*t58*t117)+t44*t82*v_om-t59*t117*v_th)-(t37*(t43*t143-t44*t54*t59*t82))/4.0+(v_th*v_ph*(t4*(Idyy+Idzz+t24+t93+t127+t129+t133)*2.0-t9*t44*t58*t99*2.0))/4.0-g*t4*t8*t103
		nle[3,0] = Tomega-rw*t8*t104*t114+t6*t10*t36*t83+t4*v_th*v_ph*(Iwyy+t121+t83*(t47*2.0-1.0))-t6*t10*t37*t43*t83
		nle[4,0] = -Tneta-(v_th*(v_ph*(t136-t139)+t59*v_om*(t18+t111)))/2.0+(v_om*v_ph*(t9*t44*t82*2.0-t4*t58*t117*2.0))/2.0+(t35*t44*t82)/2.0+(t37*t99*(t2*t9+t4*t5*t58)*(t5*t9-t2*t4*t58))/2.0-(t36*t44*t79*t82)/2.0
		nle[5,0] = t4*v_th*(t7*t84*v_ph-rf*t8*t11*v_om)*-2.0+t11*t60*t124+rf*t7*t8*t114+t9*t11*t36*t84+rw*t11*t87*v_ph
		nle[6,0] = t4*v_th*(t11*t84*v_ph+rf*t7*t8*v_om)*-2.0-t7*t60*t124+rf*t8*t11*t114-t7*t9*t36*t84-rw*t7*t87*v_ph
		nle[7,0] = t4*t35*t60+t4*t36*t84-rf*t8*t9*v_om*v_th*2.0

		# # Add baumgarte stabilization
		# err_p, err_v = self._err_posvel_contact(x[:,0])
		# nle[5:8,0] += ((2 * self.baumgarte_factor * err_v) + (self.baumgarte_factor**2 * err_p))

		acc = np.linalg.solve(M, -nle)
		acc[6,0] -= acc[5,0]

		dx = np.zeros((16, 1))
		dx[0,0] = vx
		dx[1,0] = vy
		dx[2,0] = vz
		dx[3,0] = v_ph
		dx[4,0] = v_th
		dx[5,0] = v_om
		dx[6,0] = v_ps - v_om
		dx[7,0] = v_ne

		dx[8:16,0] = acc[:,0]

		return dx
	
	def _jacobian_contact(self, x):
		# angles - [base_yaw, base_roll, base_pitch, wheel_pitch, dumbell_yaw]
		rw = self.rw
		rf = self.rf
		ph = x[3]
		th = x[4]
		om = x[5]
		ps = x[6] + om
		ne = x[7]

		J = np.zeros((3, 8))
		J[0,0] = 1.
		J[0,3] = -(np.cos(ph)*(rw + rf*np.cos(om))*np.sin(th)) + rf*np.sin(ph)*np.sin(om)
		J[0,4] = -(np.cos(th)*(rw + rf*np.cos(om))*np.sin(ph))
		J[0,5] = -(rf*np.cos(ph)*np.cos(om)) + rf*np.sin(th)*np.sin(ph)*np.sin(om)
		J[0,6] = -(rw*np.cos(ph))

		J[1,1] = 1.
		J[1,3] = -((rw + rf*np.cos(om))*np.sin(th)*np.sin(ph)) - rf*np.cos(ph)*np.sin(om)
		J[1,4] = np.cos(th)*np.cos(ph)*(rw + rf*np.cos(om))
		J[1,5] = -(rf*(np.cos(om)*np.sin(ph) + np.cos(ph)*np.sin(th)*np.sin(om)))
		J[1,6] = -(rw*np.sin(ph))

		J[2,2] = 1.
		J[2,4] = (rw + rf*np.cos(om))*np.sin(th)
		J[2,5] = rf*np.cos(th)*np.sin(om)

		# v = ... + J[:,5] * v_om + J[:,6] * (v_ps + v_om)
		# v = ... + (J[:,5] + J[:,6]) * v_om + J[:,6] * v_ps
		J[:, 5] += J[:, 6]

		return J
	
	def _jacobian_contact_trace(self, x):
		
		rw = self.rw
		rf = self.rf
		ph = x[3]
		th = x[4]
		om = x[5]
		ps = x[6] + om
		ne = x[7]

		J = np.zeros((3, 8))
		J[0,0] = 1.
		J[0,3] = -(np.cos(ph)*(rw + rf*np.cos(om))*np.sin(th)) + rf*np.sin(ph)*np.sin(om)
		J[0,4] = -(np.cos(th)*(rw + rf*np.cos(om))*np.sin(ph))
		J[0,5] = -(rf*np.cos(ph)*np.cos(om)) + rf*np.sin(th)*np.sin(ph)*np.sin(om)

		J[1,1] = 1.
		J[1,3] = -((rw + rf*np.cos(om))*np.sin(th)*np.sin(ph)) - rf*np.cos(ph)*np.sin(om)
		J[1,4] = np.cos(th)*np.cos(ph)*(rw + rf*np.cos(om))
		J[1,5] = -(rf*(np.cos(om)*np.sin(ph) + np.cos(ph)*np.sin(th)*np.sin(om)))

		J[2,2] = 1.
		J[2,4] = (rw + rf*np.cos(om))*np.sin(th)
		J[2,5] = rf*np.cos(th)*np.sin(om)

		J[:, 5] += J[:, 6]

		return J

	def _jacobian_com(self, x):

		mw = self.mw
		mf = self.mf
		md = self.md

		rw = self.rw
		rf = self.rf
		rd = self.rd

		ph = x[3]
		th = x[4]
		om = x[5]
		ps = x[6] + om
		ne = x[7]

		J = np.zeros((3, 8))
		J[0,0] = 1.
		J[0,3] = ((md*rd - mw*rf)*(np.cos(ph)*np.cos(om)*np.sin(th) - np.sin(ph)*np.sin(om)))/(md + mf + mw)
		J[0,4] = ((md*rd - mw*rf)*np.cos(th)*np.cos(om)*np.sin(ph))/(md + mf + mw)
		J[0,5] = ((md*rd - mw*rf)*(np.cos(ph)*np.cos(om) - np.sin(th)*np.sin(ph)*np.sin(om)))/(md + mf + mw)

		J[1,1] = 1.
		J[1,3] = ((md*rd - mw*rf)*(np.cos(om)*np.sin(th)*np.sin(ph) + np.cos(ph)*np.sin(om)))/(md + mf + mw)
		J[1,4] = ((-(md*rd) + mw*rf)*np.cos(th)*np.cos(ph)*np.cos(om))/(md + mf + mw)
		J[1,5] = ((md*rd - mw*rf)*(np.cos(om)*np.sin(ph) + np.cos(ph)*np.sin(th)*np.sin(om)))/(md + mf + mw)

		J[2,2] = 1.
		J[2,4] = ((-(md*rd) + mw*rf)*np.cos(om)*np.sin(th))/(md + mf + mw)
		J[2,5] = ((-(md*rd) + mw*rf)*np.cos(th)*np.sin(om))/(md + mf + mw)

		J[:, 5] += J[:, 6]

		return J

	def _err_posvel_contact(self, x):

		# Rframe = transfm.Rotation.from_euler('yxz', [x[5], x[4], x[3]]).as_matrix()
		# Rwheel = transfm.Rotation.from_euler('yxz', [0., x[4], x[3]]).as_matrix()
		# err_p_ = x[:3] + Rframe @ np.array([0., 0., -self.rf]) + Rwheel @ np.array([0., 0., -self.rw])
		# err_p_[:2] = 0.

		# c_J = self._jacobian_contact(x)
		# err_v_ = c_J @ x[8:]

		ph = x[3]
		th = x[4]
		om = x[5]

		err_p = np.zeros(3)
		err_p[2] = x[2] - np.cos(th)*(self.rw + self.rf*np.cos(om))
		
		t2 = np.cos(om)
		t3 = np.cos(ph)
		t4 = np.cos(th)
		t5 = np.sin(om)
		t6 = np.sin(ph)
		t7 = np.sin(th)
		t8 = self.rf*t2
		t9 = self.rw+t8

		err_v = x[8:11]
		v_ph = x[11]
		v_th = x[12]
		v_om = x[13]
		v_ps = x[14]

		err_v[0] += (-v_om*(t3*t9 - self.rf*t5*t6*t7) + v_ph*(self.rf*t5*t6 - t3*t7*t9)-self.rw*t3*v_ps - t4*t6*t9*v_th)
		err_v[1] += (-v_om*(t6*t9 + self.rf*t3*t5*t7) - v_ph*(self.rf*t3*t5 + t6*t7*t9)-self.rw*t6*v_ps + t3*t4*t9*v_th)
		err_v[2] += (t7*t9*v_th + self.rf*t4*t5*v_om)

		return err_p, err_v
	
	def _create_visualizer(self):
		self.viz = meshcat.Visualizer()
		# Create the unicycle geometry
		self.viz['root'].set_object(meshcat.geometry.Box([0.04, 0.04, 2*self.rf]))  # units in meters

		wheel_color = 0x282828
		wheel_radii = np.array([self.rw, 0.04, self.rw])
		wheel_reflectivity = 0.9
		self.viz['root']['wheel'].set_object(
			meshcat.geometry.Ellipsoid(radii=wheel_radii),
			meshcat.geometry.MeshLambertMaterial(color=wheel_color, reflectivity=wheel_reflectivity)
		)
		
		wheel_ori_color = 0x880808
		wheel_ori_reflectivity = 0.95
		self.viz['root']['wheel']['wheel_marker'].set_object(
			meshcat.geometry.TriangularMeshGeometry(
				vertices=np.array([[0.02, wheel_radii[1], 0.], [-0.02, wheel_radii[1], 0.], [0., wheel_radii[1], -0.9*wheel_radii[2]]]),
				faces=np.array([[0, 1, 2]])
			),
			meshcat.geometry.MeshLambertMaterial(color=wheel_ori_color, reflectivity=wheel_ori_reflectivity)
		)

		upper_body_color = 0x880808
		upper_body_reflectivity = 0.95
		self.viz['root']['upper_body'].set_object(
			meshcat.geometry.Box([0.4, 0.04, 0.04]),
			meshcat.geometry.MeshLambertMaterial(color=upper_body_color, reflectivity=upper_body_reflectivity)
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
			waypoint_pose[:3,:3] = transfm.Rotation.from_euler('yxz', [self.reference_trajectory[5,ii], self.reference_trajectory[4,ii], self.reference_trajectory[3,ii]]).as_matrix()
			self.viz['traj_point_%d'%(ii)].set_transform(waypoint_pose)

	def _update_visualizer(self):
		
		poseB = np.eye(4)
		poseB[:3,3] = self.state[:3]
		poseB[:3,:3] = transfm.Rotation.from_euler('yxz', [self.state[5],  self.state[4], self.state[3]]).as_matrix()
		self.viz['root'].set_transform(poseB)

		poseWrB = np.eye(4)
		poseWrB[:3,3] = np.array([0., 0., -self.rf])
		poseWrB[:3,:3] = transfm.Rotation.from_euler('yxz', [self.state[6], 0., 0.]).as_matrix()
		self.viz['root']['wheel'].set_transform(poseWrB)

		poseUBrB = np.eye(4)
		poseUBrB[:3,3] = np.array([0., 0., self.rd])
		poseUBrB[:3,:3] = transfm.Rotation.from_euler('yxz', [0., 0., self.state[7]]).as_matrix()
		self.viz['root']['upper_body'].set_transform(poseUBrB)

		self.viz['root'].set_cam_pos(state[:3] + np.array([-2, 0, 1]))
		self.viz['root'].set_cam_target(state[:3])

		return self.viz['root'].get_image()

	def render(self, model='human'):
		if (not (hasattr(self, 'viz') and isinstance(self.viz, meshcat.Visualizer))):
			self._create_visualizer()

		return self._update_visualizer()

	def close (self):
		pass