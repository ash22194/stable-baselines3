from gymnasium import spaces
import numpy as np
import torch as th
import meshcat
import scipy.spatial.transform as transfm


def th_get_rotation(ry, rx, rz, device=None, dtype=None):
	# Function to construct batch of rotation matrices from pitch, roll and yaw angles
	batch_size = ry.shape[0]
	assert batch_size==rx.shape[0] and batch_size==rz.shape[0], "rx, ry and rz must be of the same size"

	Rz = th.zeros((batch_size, 3,3), device=device, dtype=dtype)
	Rz[:,2,2] = 1.
	Rz[:,0,0] = th.cos(rz)
	Rz[:,0,1] = -th.sin(rz)
	Rz[:,1,0] = -Rz[:,0,1]
	Rz[:,1,1] = Rz[:,0,0]

	Rx = th.zeros((batch_size, 3,3), device=device, dtype=dtype)
	Rx[:,0,0] = 1.
	Rx[:,1,1] = th.cos(rx)
	Rx[:,1,2] = -th.sin(rx)
	Rx[:,2,1] = -Rx[:,1,2]
	Rx[:,2,2] = Rx[:,1,1]

	Ry = th.zeros((batch_size, 3,3), device=device, dtype=dtype)
	Ry[:,1,1] = 1.
	Ry[:,0,0] = th.cos(ry)
	Ry[:,0,2] = th.sin(ry)
	Ry[:,2,0] = -Ry[:,0,2]
	Ry[:,2,2] = Ry[:,0,0]

	return th.bmm(Rz, th.bmm(Rx, Ry))


class GPUUnicycle:
	"""Custom Environment that follows gym interface"""
	metadata = {'render_modes': ['human']}

	def __init__(self, device='cpu', num_envs=1, param=dict(), dt=1e-3, T=2., normalized_actions=True, normalized_observations=True, nonlinear_cost=True, alpha_cost=1., alpha_action_cost=1., alpha_terminal_cost=1.):
		# super(Unicycle, self).__init__()
		# Define model paramters
		mw = 0.5
		mf = 0.65 + mw
		md = 2.64 * 2.

		rw = 0.1524
		frame_length = 0.39
		rf = frame_length / 2.
		rd = frame_length / 2.
		upper_body_length = 0.4

		goal = np.zeros((16, 1)) # full dims, not all may be relevant
		goal[14,0] = 10

		param_ = {'mw': mw, 'mf': mf, 'md': md, 'rw': rw, 'rf': rf, 'rd': rd, \
			'Iw': np.diag([mw*(rw**2 + 0.04**2)/5, 2*mw*(rw**2)/5, mw*(rw**2 + 0.04**2)/5]),\
			'If': np.diag([mf*(frame_length**2 + 0.08**2)/12, mf*(frame_length**2 + 0.08**2)/12, 2*mf*(0.08**2)/12]),\
			'Id': np.diag([md*(upper_body_length**2 + 0.08**2)/12, md*(upper_body_length**2 + 0.08**2)/12, 2*md*(0.08**2)/12]),\
			'alpha': -np.pi/2, 'g': 9.81, 'fcoeff': 0.05, 'T': 2, 'dt':1e-3, 'gamma_':0.9995, 'X_DIMS': 16, 'U_DIMS': 2,\
			'goal': goal, 'u0': np.zeros((2,1)),\
			'Q': np.diag([0.25, 1, 0.25, 0.025, 0.0001, 0.001, 0.0005, 0.0001]), 'R': (np.eye(2) / 5000.), 'QT': 2*np.diag([1.,1.,1.,1.,1.,1.,1.,1.]),\
			# 'Q': np.diag([0.1,0.1, 0.0005,0.0005, 0.,0., 0.00025, 0.001]), 'R': (np.eye(2) / 5000.), 'QT': 10*np.eye(8), \
			# 'Q': np.diag([1.,1., 0.001,0.001, 0.01,0.0001, 0.0025, 0.0005]), 'R': (np.eye(2) / 5000.), 'QT': 2*np.eye(8), \
			'x_sample_limits': np.array([[-np.pi, np.pi], [-np.pi/6, np.pi/6], [-np.pi/6, np.pi/6], [-np.pi, np.pi], [-np.pi/3, np.pi/3], [-1., 1.], [-1., 1.], [-1., 1.], [5., 15.], [-1., 1.]]),\
			'x_bounds': np.array([[-20., 20.], [-20., 20.], [0., 2.], [-2*np.pi, 2*np.pi], [-np.pi/3, np.pi/3], [-np.pi/3, np.pi/3], [-10*np.pi, 10*np.pi], [-4*np.pi/3, 4*np.pi/3], [-8, 8], [-8, 8], [-8, 8], [-8., 8.], [-8., 8.], [-8., 8.], [-5., 25.], [-8., 8.]]),\
			'u_limits': np.array([[-15., 15.], [-15., 15.]])}
		param_.update(param)
		param_.update({'dt':dt, 'T':T})
		param_['lambda_'] = (1. - param_['gamma_']) / param_['dt']
		param.update(param_)
		self.device = device
		self.np_dtype = np.float32
		self.th_dtype = th.float32
		self.num_envs = num_envs

		self.X_DIMS = param['X_DIMS'] # dimension of observations
		self.independent_dims = np.array([3,4,5,6,7,11,12,13,14,15]) # the same length as x_sample_limits
		self.observation_dims = np.array([3,4,5,7,11,12,13,14,15])
		self.cost_dims = np.array([4,5,7,11,12,13,14,15])
		self.U_DIMS = param['U_DIMS']
		self.goal = param['goal']
		self.u0 = param['u0']
		self.T = param['T']
		self.dt = param['dt']
		self.horizon = round(self.T / self.dt)
		self.x_sample_limits = param['x_sample_limits'] # reset within these limits
		self.x_bounds = param['x_bounds']
		self.u_limits = param['u_limits']
		self.Q = param['Q']
		self.QT = param['QT']
		self.R = param['R']
		self.gamma_ = param['gamma_']
		self.lambda_ = param['lambda_']

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
		self.baumgarte_factor = 10
		self.normalized_actions = normalized_actions
		self.normalized_observations = normalized_observations
		self.nonlinear_cost = nonlinear_cost
		self.alpha_cost = alpha_cost
		# self.alpha_action_cost = alpha_action_cost
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
		self.th_x_bounds = th.asarray(param['x_bounds'], device=device, dtype=self.th_dtype)
		self.th_u_limits = th.asarray(param['u_limits'], device=device, dtype=self.th_dtype)
		self.th_Q = th.asarray(param['Q'], device=device, dtype=self.th_dtype)
		self.th_QT = th.asarray(param['QT'], device=device, dtype=self.th_dtype)
		self.th_goal = th.asarray(param['goal'][:,0], device=device, dtype=self.th_dtype)
		self.th_R = th.asarray(param['R'], device=device, dtype=self.th_dtype)
		self.th_u0 = th.asarray(param['u0'][:,0], device=device, dtype=self.th_dtype)

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
		self.obs = th.zeros((self.num_envs, self.observation_dims.shape[0]+1), device=self.device, dtype=self.th_dtype)
		self.cumm_reward = th.zeros((self.num_envs, ), device=self.device, dtype=self.th_dtype)

	def step(self, action):
		# If scaling actions use this
		if (self.normalized_actions):
			action = 0.5*((self.th_u_limits[:,0] + self.th_u_limits[:,1]) + action*(self.th_u_limits[:,1] - self.th_u_limits[:,0]))

		state_ = self.dyn_rk4(self.state, action, self.dt)
		state_ = self._enforce_bounds(state_)

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
			self.state[:,self.independent_dims] = state[:,self.independent_dims]
			self._enforce_bounds(self.state)
			self.step_count[:] = 0.
			self.cumm_reward[:] = 0.
		
		else:
			if (done is not None):
				assert done.shape[0]==self.num_envs and type(done)==th.Tensor and done.dtype==th.bool, "done tensor of shape %d and type %s"%(done.shape[0], type(done))
			else:
				done = th.ones(self.num_envs, device=self.device, dtype=bool)
			num_done = th.sum(done)
			if (num_done > 0):
				step_count_new = 0*th.randint(low=0, high=self.horizon, size=(num_done,1), dtype=self.th_dtype, device=self.device)
				state_new = th.zeros((num_done, self.X_DIMS), device=self.device, dtype=self.th_dtype)
				state_new[:,self.independent_dims] = ((th.rand((num_done, self.independent_dims.shape[0]), device=self.device, dtype=self.th_dtype) - 0.5) * (self.th_x_sample_limits_range))
				state_new = (((self.horizon - step_count_new) / self.horizon) * state_new)
				state_new[:,self.independent_dims] += self.th_x_sample_limits_mid
				state_new = self._enforce_bounds(state_new)

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
		return th.linalg.norm((self.state - self.th_goal)[:,self.cost_dims], dim=1, keepdim=False)

	def _set_obs(self):
		self.obs[:,:self.observation_dims.shape[0]] = self.state[:,self.observation_dims]
		self.obs[:,-1] = self.step_count
	
	def _get_cost(self, action, state_):
		if (self.nonlinear_cost):
			y = self.get_taskspace_obs()
		else:
			y = (self.state - self.th_goal)[:,self.cost_dims]

		action = th.asarray(action, dtype=self.th_dtype)
		cost = th.sum((y @ self.th_Q) * y, dim=1, keepdim=False) * self.alpha_cost 
		cost = cost + th.sum(((action - self.th_u0) @ self.th_R) * (action - self.th_u0), dim=1, keepdim=False) * self.alpha_action_cost
		cost = cost * self.dt

		reached_goal = th.linalg.norm((state_ - self.th_goal)[:,self.cost_dims], dim=1, keepdim=False) <= 1e-2

		return cost, reached_goal
	
	def _get_terminal_cost(self, done=None):
		if (done is not None):
			assert done.shape[0]==self.num_envs and type(done)==th.Tensor and done.dtype==th.bool, "done tensor of shape %d and type %s"%(done.shape[0], type(done))
		else:
			done = th.ones(self.num_envs, device=self.device, dtype=bool)
		
		if (self.nonlinear_cost):
			y = self.get_taskspace_obs()
		else:
			y = (self.state - self.th_goal)[:,self.cost_dims]
		cost = th.sum((y @ self.th_QT) * y, dim=1, keepdim=False) * self.alpha_terminal_cost
		cost = cost * done
		
		return cost
	
	def get_taskspace_obs(self):
		Rframe = th_get_rotation(self.state[:,5], self.state[:,4], self.state[:,3], device=self.device, dtype=self.th_dtype)

		p_wheel = Rframe @ th.asarray([0.,0.,-self.rf], device=self.device, dtype=self.th_dtype)
		p_dumbell = Rframe @ th.asarray([0.,0.,self.rd], device=self.device, dtype=self.th_dtype)

		p_com = (self.mw*p_wheel + self.md*p_dumbell) / (self.mf + self.mw + self.md)
		v_com = th.squeeze(th.bmm(self._jacobian_com(self.state), th.unsqueeze(self.state[:,8:16], dim=2)), dim=2)

		# ry_wheel = self.state[:,5]+self.state[:,6]
		# p_wheel_rel = th.zeros((self.num_envs, 3, 1), device=self.device, dtype=self.th_dtype)
		# p_wheel_rel[:,0,0] = self.rw*th.sin(ry_wheel)
		# p_wheel_rel[:,2,0] = -self.rw*th.cos(ry_wheel)
		# Rwheel = th_get_rotation(ry_wheel, self.state[:,4], self.state[:,3], device=self.device, dtype=self.th_dtype)
		# p_con = p_wheel + th.squeeze(th.bmm(Rwheel, p_wheel_rel))

		ry_wheel = th.zeros(self.num_envs, device=self.device, dtype=self.th_dtype)
		p_wheel_rel = th.asarray([0., 0., -self.rw], device=self.device, dtype=self.th_dtype)
		Rwheel = th_get_rotation(ry_wheel, self.state[:,4], self.state[:,3], device=self.device, dtype=self.th_dtype)
		p_con = p_wheel + Rwheel @ p_wheel_rel

		v_con = th.squeeze(th.bmm(self._jacobian_contact_trace(self.state), th.unsqueeze(self.state[:,8:16], dim=2)), dim=2)

		y = th.zeros((self.num_envs, 8), device=self.device, dtype=self.th_dtype)
		y[:,0] = p_com[:,0] - p_con[:,0]
		y[:,1] = p_com[:,1] - p_con[:,1]
		y[:,2] = v_com[:,0] - v_con[:,0]
		y[:,3] = v_com[:,1] - v_con[:,1]
		y[:,4] = self.state[:,7] - self.th_goal[7]
		y[:,5] = self.state[:,15] - self.th_goal[15]
		y[:,6] = self.state[:,11] - self.th_goal[11]
		y[:,7] = self.state[:,14] - self.th_goal[14]

		return y
	
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

		ph = x[:,3]
		theta = x[:,4]
		om = x[:,5]
		ps = x[:,6] + om
		ne = x[:,7]

		vx = x[:,8]
		vy = x[:,9]
		vz = x[:,10]
		v_ph = x[:,11]
		v_th = x[:,12]
		v_om = x[:,13]
		v_ps = x[:,14] + v_om
		v_ne = x[:,15]

		Tomega = -u[:,0]
		Tneta = u[:,1]

		t2 = th.cos(ne)
		t3 = th.cos(om)
		t4 = th.cos(theta)
		t5 = th.sin(ne)
		t6 = th.cos(ps)
		t7 = th.cos(ph)
		t8 = th.sin(om)
		t9 = th.sin(theta)
		t10 = th.sin(ps)
		t11 = th.sin(ph)
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
		t34 = theta*2.0
		t35 = v_om**2
		t36 = v_th**2
		t37 = v_ph**2
		t38 = ps*2.0
		t57 = v_om*v_ph*2.0
		t63 = -Idyy
		t64 = -Idzz
		t70 = -Iwzz
		t39 = th.cos(t32)
		t40 = t2**2
		t41 = th.cos(t33)
		t42 = th.cos(t34)
		t43 = t4**2
		t44 = th.sin(t32)
		t45 = t5**2
		t46 = th.cos(t38)
		t47 = t6**2
		t48 = t14*2.0
		t49 = t14*4.0
		t50 = t16*2.0
		t51 = t16*4.0
		t52 = th.sin(t33)
		t53 = t8**2
		t54 = th.sin(t34)
		t55 = t9**2
		t56 = th.sin(t38)
		t58 = th.cos(t12)
		t59 = th.sin(t12)
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
		t97 = th.cos(t95)
		t98 = th.sin(t95)
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

		M = th.zeros((x.shape[0],8,8), device=self.device, dtype=self.th_dtype)

		M[:,0,0] = -t8*t11*t103+t7*t9*t130
		M[:,1,0] = t4*t11*t131
		M[:,2,0] = t103*(t3*t7-t8*t9*t11)
		M[:,3,0] = t7*t85
		M[:,5,0] = 1.0

		M[:,0,1] = t7*t8*t103+t9*t11*t130
		M[:,1,1] = -t4*t7*t131
		M[:,2,1] = t103*t119
		M[:,3,1] = t11*t85
		M[:,6,1] = 1.0

		M[:,1,2] = -t9*t131
		M[:,2,2] = -t4*t8*t103
		M[:,7,2] = 1.0

		M[:,0,3] = t53*t86+(t55*(Idyy+Idzz+t24+t28+t86+t110+t121+t41*t86))/2.0+t43*(t90*(t88+t89)+Idxx*t79+Ifxx*t53+Iwzz*t47+Ifzz*t3**2+Iwxx*t10**2)+(t44*t54*t58*t82)/2.0
		M[:,1,3] = t4*(t116+t143+rw*t8*t109)*(-1.0/4.0)+t2*t5*t9*t59*t82
		M[:,2,3] = t132+(t9*(Idyy+Idzz+t24+t93+t110))/2.0
		M[:,3,3] = t9*(Iwyy+t118)
		M[:,4,3] = t105
		M[:,5,3] = rf*t8*t11-t7*t9*t84
		M[:,6,3] = -rf*t7*t8-t9*t11*t84

		M[:,0,4] = t9*t128-(t4*(t56*t83+t52*(Ifxx-Ifzz+t86)-t98*(-Idxx+t88+t89)))/2.0
		M[:,1,4] = Idxx/2.0+Idyy/4.0+Idzz/4.0+Ifxx/2.0+Ifzz/2.0+Iwxx/2.0+Iwzz/2.0+t86/2.0+t115/4.0+t122/4.0-t125/4.0+t133/4.0-(t39*t79*t99)/4.0
		M[:,2,4] = t128
		M[:,3,4] = 0.0
		M[:,4,4] = t81
		M[:,5,4] = -t4*t11*t84
		M[:,6,4] = t4*t7*t84
		M[:,7,4] = t9*t84

		M[:,0,5] = t132+(t9*(Idyy+Idzz+t24+t93+t110+t121))/2.0
		M[:,1,5] = t128
		M[:,2,5] = Idyy/2.0+Idzz/2.0+Ifyy+t86+t110/2.0
		M[:,3,5] = t118
		M[:,5,5] = -t7*t60+rf*t8*t9*t11
		M[:,6,5] = -rf*t119
		M[:,7,5] = rf*t4*t8

		M[:,0,6] = Iwyy*t9
		M[:,3,6] = Iwyy
		M[:,5,6] = -rw*t7
		M[:,6,6] = -rw*t11

		M[:,0,7] = t105
		M[:,1,7] = t81
		M[:,4,7] = Idxx

		t139 = t9*t59*t126
		et1 = t35*(rw*t8*t9*t104*4.0+t2*t4*t5*t59*t82*4.0)*(-1.0/4.0)-(v_ne*(Idxx*(t4*t58*v_om-t9*t59*v_th)*8.0-t39*t100*(t9*t59*v_th*2.0+t4*t58*(t72*2.0+v_om)*2.0)+t44*t82*(t9*v_om*8.0+v_ph*(t79*4.0-t42*t107*2.0)-t4*t98*v_th*4.0)))/8.0+(t36*(t9*(t116+t143)+t4*t44*t59*t99))/4.0+(v_th*(v_ph*(t135+t54*(t25+t29+t67+t68+t69+t71+t93+t94+t115+t122+t127+t133+t107*t110))+t4*(v_om*(Idyy+Idzz+t24+t93+t125-t133+t39*t79*t99)+v_ps*(t28-t46*t83*2.0))*2.0))/4.0+v_ph*(fcoeff-rw*t8*t72*t104+t43*t56*t83*v_ps)
		et2 = v_om*v_ph*(t43*t143*-2.0+rw*t8*t55*t109*2.0+t44*t54*t59*t82*2.0)*(-1.0/4.0)

		nle = th.zeros((x.shape[0],8,1), device=self.device, dtype=self.th_dtype)
		nle[:,0,0] = et1+et2
		nle[:,1,0] = t37*(t135+t54*(t21+t25+t29+t62+t63+t67+t68+t69+t71+t93+t115+t122+t133+t18*t97-t89*t97*2.0+t40*(Idyy+t64*t90)*4.0+t39*(Idyy+t65)))*(-1.0/8.0)-(v_ph*(v_om*(t4*(Idyy+Idzz+t24+t93+t122+t127+t129+t133)-t9*t44*t58*t99)+t4*v_ps*(Iwyy+t46*t83)*2.0))/2.0-(v_th*(v_om*(t134*2.0-t98*(t94-t110)*2.0+rw*t8*t109*2.0)+t56*v_ps*(Iwxx*4.0-Iwzz*4.0)))/4.0+(v_ne*(v_ph*(t136-t139)-t59*t126*v_om+t44*t79*t99*v_th))/2.0-g*t9*t131+(t35*t44*t58*t100)/8.0
		nle[:,2,0] = -Tomega+(t36*t143)/4.0-v_ne*(v_ph*(t9*t44*t82-t4*t58*t117)+t44*t82*v_om-t59*t117*v_th)-(t37*(t43*t143-t44*t54*t59*t82))/4.0+(v_th*v_ph*(t4*(Idyy+Idzz+t24+t93+t127+t129+t133)*2.0-t9*t44*t58*t99*2.0))/4.0-g*t4*t8*t103
		nle[:,3,0] = Tomega-rw*t8*t104*t114+t6*t10*t36*t83+t4*v_th*v_ph*(Iwyy+t121+t83*(t47*2.0-1.0))-t6*t10*t37*t43*t83
		nle[:,4,0] = -Tneta-(v_th*(v_ph*(t136-t139)+t59*v_om*(t18+t111)))/2.0+(v_om*v_ph*(t9*t44*t82*2.0-t4*t58*t117*2.0))/2.0+(t35*t44*t82)/2.0+(t37*t99*(t2*t9+t4*t5*t58)*(t5*t9-t2*t4*t58))/2.0-(t36*t44*t79*t82)/2.0
		nle[:,5,0] = t4*v_th*(t7*t84*v_ph-rf*t8*t11*v_om)*-2.0+t11*t60*t124+rf*t7*t8*t114+t9*t11*t36*t84+rw*t11*t87*v_ph
		nle[:,6,0] = t4*v_th*(t11*t84*v_ph+rf*t7*t8*v_om)*-2.0-t7*t60*t124+rf*t8*t11*t114-t7*t9*t36*t84-rw*t7*t87*v_ph
		nle[:,7,0] = t4*t35*t60+t4*t36*t84-rf*t8*t9*v_om*v_th*2.0

		# # Add baumgarte stabilization
		# err_p, err_v = self._err_posvel_contact(x)
		# nle[:,5:8,0] += ((2 * self.baumgarte_factor * err_v) + (self.baumgarte_factor**2 * err_p))

		acc = th.squeeze(th.linalg.solve(M, -nle), dim=2)
		acc[:,6] -= acc[:,5]

		dx = th.zeros((x.shape[0], 16), device=self.device, dtype=self.th_dtype)
		dx[:,0] = vx
		dx[:,1] = vy
		dx[:,2] = vz
		dx[:,3] = v_ph
		dx[:,4] = v_th
		dx[:,5] = v_om
		dx[:,6] = v_ps - v_om
		dx[:,7] = v_ne

		dx[:,8:16] = acc

		return dx
	
	def dyn_full_ana(self, x, u):

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
		al = self.alpha
		fcoeff = self.fcoeff

		# x = [x, y, z, ph, theta, om, turntable, vx, vy, vz, dph, dth, dom, dwheel, dturntable]
		
		ph = x[:,3]
		theta = x[:,4]
		om = x[:,5]
		ps = x[:,6] + om
		ne = x[:,7]

		vx = x[:,8]
		vy = x[:,9]
		vz = x[:,10]
		v_ph = x[:,11]
		v_th = x[:,12]
		v_om = x[:,13]
		v_ps = x[:,14] + v_om
		v_ne = x[:,15]

		Tomega = -u[:,0]
		Tneta = u[:,1]

		t2 = th.cos(ne)
		t3 = th.cos(om)
		t4 = th.cos(ph)
		t5 = th.cos(ps)
		t6 = th.cos(theta)
		t7 = th.sin(ne)
		t8 = th.sin(om)
		t9 = th.sin(ph)
		t10 = th.sin(ps)
		t11 = th.sin(theta)
		t12 = al+om
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
		t27 = Idxx**2
		t28 = Iwxx*2.0
		t29 = Iwxx*4.0
		t30 = Iwyy*2.0
		t31 = Iwyy*4.0
		t32 = Iwzz*2.0
		t33 = Iwzz*4.0
		t34 = Tneta*2.0
		t35 = Tomega*4.0
		t37 = ne*2.0
		t38 = om*2.0
		t39 = ps*2.0
		t41 = rw**2
		t42 = theta*2.0
		t43 = v_om**2
		t44 = v_ph**2
		t45 = v_th**2
		t70 = v_om*v_ph*2.0
		t76 = -Idxx
		t78 = -Idyy
		t79 = -Idzz
		t83 = -Ifzz
		t86 = -Iwzz
		t99 = Idxx/2.0
		t100 = Idyy/2.0
		t101 = Idyy/4.0
		t102 = Idzz/2.0
		t103 = Idzz/4.0
		t104 = Ifxx/2.0
		t105 = Ifzz/2.0
		t106 = Iwxx/2.0
		t107 = Iwzz/2.0
		t46 = th.cos(t37)
		t47 = t2**2
		t48 = th.cos(t38)
		t49 = t3**2
		t50 = t4**2
		t51 = t4**3
		t52 = th.cos(t39)
		t53 = t5**2
		t54 = th.cos(t42)
		t55 = t6**2
		t56 = t14*2.0
		t57 = t14*4.0
		t58 = t16*2.0
		t59 = t16*4.0
		t60 = th.sin(t37)
		t61 = t7**2
		t62 = th.sin(t38)
		t63 = t8**2
		t64 = t9**2
		t65 = t9**3
		t66 = th.sin(t39)
		t67 = t10**2
		t68 = th.sin(t42)
		t69 = t11**2
		t71 = Iwyy*t11
		t72 = th.cos(t12)
		t73 = th.sin(t12)
		t74 = rf*t3
		t75 = mw+t13
		t77 = -t18
		t80 = -t21
		t81 = -t22
		t82 = -t23
		t84 = -t26
		t85 = -t28
		t87 = -t32
		t88 = -t33
		t89 = t11*v_ph
		t90 = t3*t4
		t91 = rf*t13
		t92 = md*t17
		t93 = t3*t9
		t94 = t2*t11
		t95 = t7*t11
		t96 = -t16
		t111 = t11*t70
		t112 = Idyy+t79
		t114 = Iwxx+t86
		t117 = t14*t17
		t118 = rw*t3*t14
		t119 = rw*t3*t16
		t120 = rf*t4*t8
		t122 = rf*t8*t9
		t124 = t11*v_om*8.0
		t132 = t43+t44
		t133 = t8*t9*t11
		t141 = t12*2.0
		t142 = t4*t8*t11
		t144 = Iwyy*t4*t8*t14
		t146 = Iwyy*t4*t8*t15
		t148 = Iwyy*t8*t9*t14
		t150 = Iwyy*t8*t9*t15
		t173 = rf*t8*t11*v_om*v_th*2.0
		t97 = -t58
		t98 = -t59
		t108 = t89*2.0
		t109 = Ifyy*t71
		t110 = t73**2
		t113 = Idyy+t80
		t115 = rw+t74
		t116 = rw*t75
		t121 = t89+v_ps
		t123 = t60**2
		t125 = Idzz*t47
		t126 = Ifzz*t49
		t127 = Iwzz*t53
		t128 = Idyy*t61
		t129 = Ifxx*t63
		t130 = Iwxx*t67
		t131 = t72**2
		t134 = t122*v_om
		t135 = t17*t56
		t137 = Idyy+Idzz+t77
		t138 = Iwyy*t120
		t139 = Iwyy*t122
		t140 = rf*t14*t71
		t143 = t120*v_om
		t147 = t6*t72*v_om
		t151 = th.cos(t141)
		t152 = t11*t73*v_th
		t153 = th.sin(t141)
		t155 = rw*t3*t96
		t156 = -t122
		t159 = t19+t80
		t160 = t20+t81
		t161 = t112**2
		t162 = t28+t87
		t163 = t29+t88
		t164 = rd*t14*t71
		t165 = t71*t100
		t166 = t71*t102
		t167 = -t133
		t169 = t6*t43*t74
		t171 = Idxx*t71*t72
		t172 = t117/2.0
		t174 = t2*t6*t72
		t175 = t4*t71*t74
		t176 = t14+t91
		t177 = t6*t7*t72
		t178 = t15+t92
		t179 = t9*t71*t74
		t180 = t14+t96
		t181 = t14*t71*t90
		t183 = t15*t71*t90
		t185 = t14*t71*t93
		t187 = t15*t71*t93
		t189 = t11*t132
		t191 = -t173
		t194 = t48*t117
		t195 = md*rw*t50*t74
		t196 = rw*t3*t15*t50
		t197 = t50*t119
		t198 = t63*t117
		t199 = md*rw*t64*t74
		t200 = rw*t3*t15*t64
		t201 = t64*t119
		t204 = Idxx*t72*t120
		t205 = Idxx*rw*t4*t11*t72
		t207 = Idxx*rw*t9*t11*t72
		t214 = t46*t112
		t216 = t52*t114
		t217 = t66*t114
		t221 = t41*t50*t75
		t225 = t41*t64*t75
		t226 = md*rf*t8*t41*t51
		t227 = t8*t15*t41*t51
		t228 = t8*t16*t41*t51
		t229 = md*rf*t8*t41*t65
		t230 = t8*t15*t41*t65
		t231 = t8*t16*t41*t65
		t233 = t60*t112*v_om
		t235 = t11*t14*t41*t90
		t237 = t11*t14*t41*t93
		t238 = t11*t16*t41*t93
		t239 = Ifxx+t83+t117
		t244 = Idzz*t46*t71*(-1.0/2.0)
		t247 = Idxx*t4*t11*t72*t74
		t248 = t111+t132
		t250 = Idxx*t9*t11*t72*t74
		t252 = t93+t142
		t253 = t72*t76*t122
		t258 = Idxx*rw*t9*t55*t73
		t259 = t11*t60*t112
		t262 = md*t41*t64*t120
		t263 = md*t41*t50*t122
		t264 = t4*t8*t15*t41*t64
		t265 = t8*t9*t15*t41*t50
		t266 = t4*t8*t16*t41*t64
		t267 = t8*t9*t16*t41*t50
		t269 = t5*t10*t45*t114
		t276 = t11*t41*t90*t96
		t283 = t43*t60*t112
		t296 = Idxx*t9*t55*t73*t74
		t297 = rw*t4*t55*t73*t76
		t310 = Iwyy*t6*t60*t72*t100
		t320 = t6*t60*t72*t112
		t324 = t4*t55*t73*t74*t76
		t325 = Idzz*Iwyy*t6*t60*t72*(-1.0/2.0)
		t335 = t5*t10*t44*t55*t114
		t338 = (t60*t73*t112)/2.0
		t340 = t2*t6*t7*t73*t112*4.0
		t356 = t60*t68*t73*t112
		t396 = (t60*t68*t72*t112)/2.0
		t136 = t108+v_om
		t145 = md*t138
		t149 = md*t139
		t154 = t110*4.0
		t158 = Idxx*t110
		t170 = t152*2.0
		t182 = md*t175
		t184 = t4*t115*v_ph
		t186 = md*t179
		t188 = t9*t115*v_ph
		t190 = -t152
		t192 = t79*t131
		t193 = t151+3.0
		t202 = t56+t97
		t203 = t57+t98
		t208 = t4*t11*t115
		t209 = -t174
		t210 = t9*t11*t115
		t211 = -t175
		t212 = rw*t4*t121*v_ph
		t213 = rw*t9*t121*v_ph
		t215 = t46*t113
		t218 = t18*t151
		t219 = t46*t165
		t222 = -t181
		t224 = -t183
		t234 = t6*t45*t115
		t240 = t3*t176
		t241 = t3*t178
		t242 = t214*2.0
		t243 = t216*2.0
		t245 = t6*t153*v_th*4.0
		t246 = t125+t128
		t249 = t52*t162
		t251 = t66*t162
		t254 = rw*t3*t180
		t255 = Idxx+t214
		t256 = Iwyy+t216
		t277 = -t237
		t278 = t4*t6*t71*t115
		t279 = Iwyy+t221
		t280 = t6*t9*t71*t115
		t281 = Iwyy+t225
		t282 = t4*t8*t178
		t284 = t66*t163*v_ps
		t285 = t8*t9*t178
		t286 = t259*2.0
		t287 = t214/2.0
		t288 = t94+t177
		t290 = t23+t84+t135
		t293 = t90+t167
		t298 = t252**2
		t299 = -t259
		t301 = t70+t189
		t302 = g*t6*t8*t178*4.0
		t303 = t128*t151*-2.0
		t306 = t137*t151
		t307 = t4*t116*t171
		t309 = rw*t8*t11*t180*4.0
		t312 = t55*t217*v_ps
		t317 = rw*t8*t89*t180
		t321 = t62*t239
		t322 = t73*t259
		t333 = t9*t71*t72*t76*t116
		t344 = t46*t131*t159
		t345 = t46*t110*t159
		t351 = t120*t248
		t352 = t11*t60*t72*t159
		t353 = t6*t60*t73*t159
		t357 = t11*t60*t73*t159
		t360 = t320/2.0
		t361 = -t335
		t366 = t43*t60*t72*t160
		t368 = t60*t110*t159*v_th
		t369 = t45*t60*t110*t112
		t372 = t356*2.0
		t374 = t54*t60*t72*t160
		t375 = t6*t60*t112*t153
		t385 = t156*t248
		t388 = t4*t6*t115*t225
		t389 = t6*t9*t115*t221
		t390 = rf*t9*t116*t252
		t397 = t6*t8*t171*t178
		t399 = t6*t60*t99*t110*t112
		t409 = Idyy+Idzz+t24+t135+t214
		t412 = (t110*t123*t161)/4.0
		t413 = Iwyy*t178*t252
		t424 = t9*t116*t338
		t430 = t9*t60*t73*t112*t116*(-1.0/2.0)
		t431 = t71*t178*t252
		t433 = rw*t8*t180*t248
		t442 = t4*t6*t115*t338
		t448 = t221*t338
		t449 = t4*t60*t71*t73*t112*t116*(-1.0/2.0)
		t452 = t6*t9*t60*t73*t112*t115*(-1.0/2.0)
		t458 = t4*t116*t178*t252
		t459 = t6*t8*t178*t338
		t467 = t6*t8*t60*t73*t112*t178*(-1.0/2.0)
		t469 = rf*t252*t338
		t472 = -t153*(t137-t214)
		t474 = t4*t6*t115*t178*t252
		t475 = t153*(t137-t214)*-2.0
		t516 = t178*t252*t338
		t639 = t205+t247+t253+t258+t296
		t653 = t204+t207+t250+t297+t324
		t695 = t118+t155+t195+t196+t197+t199+t200+t201
		t223 = -t182
		t232 = Idyy+t192
		t260 = -t242
		t261 = -t243
		t268 = -t245
		t270 = t45*t208
		t272 = t45*t210
		t273 = rw*t3*t202
		t274 = rw*t3*t203
		t275 = rw*t8*t203
		t289 = Iwyy+t254
		t292 = t6*t72*t136*2.0
		t304 = t249/4.0
		t305 = t54*t193*2.0
		t316 = t18+t242
		t318 = t95+t209
		t326 = t76+t246
		t327 = t293**2
		t330 = t6*t256*v_ps*2.0
		t331 = -t306
		t336 = t120+t210
		t337 = t143+t188
		t341 = -t317
		t342 = t147+t190
		t343 = t73*t255*v_th
		t346 = t306/4.0
		t347 = t73*t299
		t348 = t116+t240
		t349 = t116+t241
		t359 = t6*t72*t255
		t362 = t131*t246
		t363 = t322/2.0
		t367 = t48*t290
		t370 = t352*2.0
		t371 = t62*t290
		t376 = -t345
		t378 = t4*t74*t301
		t380 = t9*t74*t301
		t381 = t193*t214
		t391 = t345/4.0
		t393 = -t366
		t398 = t357/4.0
		t410 = rf*t4*t116*t293
		t419 = t4*t6*t115*t279
		t421 = t6*t9*t115*t281
		t425 = -t412
		t427 = t6*v_th*(t134-t184)*2.0
		t428 = rf*t178*t298
		t432 = t225+t279
		t436 = t11*t409
		t445 = -t433
		t446 = -t4*t116*(t122-t208)
		t450 = t71*t430
		t451 = t6*t116*t171*t282
		t454 = rf*t252*t279
		t455 = t4*t116*t399
		t456 = t9*t116*t399
		t460 = rf*t281*t293
		t465 = t6*t71*t72*t76*t116*t285
		t470 = t9*t116*t178*t293
		t471 = t169+t191+t234
		t480 = (t4*t116*t409)/2.0
		t481 = (t9*t116*t409)/2.0
		t482 = rf*t60*t73*t112*t293*(-1.0/2.0)
		t483 = -t474
		t484 = rw*t180*t241*t252
		t486 = t6*t9*t115*t178*t293
		t487 = t309+t340
		t488 = t254+t390
		t495 = rw*t180*t241*t293
		t502 = (t4*t6*t115*t409)/2.0
		t504 = (t6*t9*t115*t409)/2.0
		t531 = rf*t293*t430
		t537 = t60*t73*t112*t178*t293*(-1.0/2.0)
		t545 = -t178*t293*(t122-t208)
		t551 = t424*(t122-t208)
		t585 = t144+t145+t146+t185+t186+t187
		t792 = t6*t8*t11*t115*t178*t695
		t901 = t138+t179+t226+t227+t228+t238+t262+t264+t266+t277
		t907 = t139+t211+t229+t230+t231+t235+t263+t265+t267+t276
		t294 = t275*2.0
		t319 = -t305
		t323 = t274/4.0
		t328 = t18+t260
		t329 = t30+t261
		t334 = t11*t289
		t339 = t47*t232*4.0
		t358 = -t346
		t373 = Iwyy*t69*t289
		t377 = t359*2.0
		t379 = -t363
		t384 = t371*2.0
		t392 = t73*t316*v_om
		t394 = -t367
		t402 = t6*t337*v_th*2.0
		t404 = -t391
		t406 = Idxx*t342*8.0
		t407 = t367/4.0
		t408 = -t398
		t414 = t4*t11*t348
		t416 = t9*t11*t348
		t417 = Iwyy*t4*t6*t349
		t418 = t170+t292
		t422 = g*t11*t349*8.0
		t429 = t153*t326
		t434 = t9*t116*t336
		t435 = t256+t273
		t437 = t4*t6*t71*t349
		t438 = rf*t178*t327
		t457 = t436/2.0
		t463 = t6*t50*t116*t349
		t464 = t6*t64*t116*t349
		t473 = t4*t55*t73*t76*t349
		t476 = t55*t64*t115*t349
		t478 = -t470
		t479 = t50*t55*t115*t349
		t485 = t289*t363
		t489 = fcoeff+t312+t341
		t490 = -t481
		t492 = rw*t6*t90*t180*t349
		t493 = rw*t6*t93*t180*t349

		t496 = -t484
		t497 = t55*t282*t349
		t498 = t55*t285*t349
		t499 = t254+t410
		t503 = t273+t409
		t505 = rf*t4*t6*t252*t349
		t509 = -t495
		t511 = t43*t487
		t512 = t27*t55*t110*t432
		t517 = rf*t6*t9*t293*t349
		t518 = rw*t4*t488
		t521 = t178*t252*t336
		t528 = t4*t6*t338*t349
		t529 = t6*t9*t60*t73*t112*t349*(-1.0/2.0)
		t530 = t4*t6*t336*t349
		t540 = t11*t178*t252*t349
		t541 = t44*t159*t288*t318
		t544 = t320+t436
		t550 = rf*t55*t63*t178*t432
		t552 = t11*t178*t293*t349
		t554 = t6*t9*t178*t252*t349
		t560 = t388+t419
		t561 = t4*t6*t178*t293*t349
		t562 = t389+t421
		t568 = Idyy+Idzz+t24+t30+t117+t194+t214+t273
		t570 = (t4*t6*t349*t409)/2.0
		t571 = (t6*t9*t349*t409)/2.0
		t575 = t126+t127+t129+t130+t158+t362
		t576 = t371+t472
		t577 = t8*t27*t55*t72*t73*t178*t432
		t583 = t338+t483
		t584 = t338+t486
		t599 = Idxx*t6*t73*t349*t436*(-1.0/2.0)
		t603 = Ifyy+t100+t102+t117+t287+t428
		t619 = t148+t149+t150+t222+t223+t224
		t662 = Idyy+Idzz+t24+t135+t331+t344+t367
		t666 = rf*t252*t585
		t669 = t469+t502
		t688 = t482+t504
		t689 = t336*t585
		t697 = v_ne*(t233-t343+v_ph*(t259-t359))*4.0
		t771 = t213+t272+t351+t380+t427
		t798 = -t792
		t869 = Idyy+Idzz+t18+t23+t26+t28+t32+t135+t249+t274+t331+t367+t376
		t904 = t25+t31+t82+t84+t85+t87+t135+t137+t249+t274+t331+t367+t381
		t959 = rf*t55*t63*t178*t901
		t967 = rf*t55*t63*t178*t907
		t350 = t329*v_ps
		t355 = t69*t294
		t400 = -t373
		t405 = t73*t328*v_om
		t415 = t11*t73*t328
		t426 = t154+t319
		t439 = -t429
		t468 = t6*t435*v_ph*v_th
		t500 = t46*t160*t418
		t507 = t282+t416
		t508 = t489*v_ph*4.0
		t514 = t178*t252*t334
		t515 = -t505
		t519 = -v_om*v_ph*(t286-t377)
		t524 = t11*t503
		t525 = t4*t6*t334*t349
		t526 = t6*t9*t334*t349
		t527 = -t511
		t532 = t178*t293*t334
		t535 = rw*t9*t499
		t536 = -t518
		t548 = (t334*t409)/2.0
		t549 = t334+t434
		t553 = -t541
		t557 = t334+t446
		t563 = t349*t457
		t569 = t9*t116*(t285-t414)
		t578 = Idxx*t6*t73*t552
		t579 = t6*t73*t76*t540
		t580 = rf*t6*t8*t560
		t581 = rf*t6*t8*t562
		t588 = (t4*t116*t544)/2.0
		t589 = Iwyy*t584
		t591 = (t9*t116*t544)/2.0
		t596 = t45*t576
		t600 = t71*t584
		t601 = (t69*t568)/2.0
		t602 = t55*t575
		t604 = t55*t576
		t608 = t60*t73*t112*(t285-t414)*(-1.0/2.0)
		t610 = t338+t517
		t614 = t338*(t285-t414)
		t615 = Ifyy+t100+t102+t117+t287+t438
		t618 = (t254*t544)/2.0
		t622 = Idxx*t6*t73*t583
		t623 = Idxx*t6*t73*t584
		t624 = t424+t492
		t628 = (t60*t73*t112*t544)/4.0
		t630 = t463+t464
		t637 = t458+t478
		t638 = -t4*t6*t349*(t285-t414)
		t642 = t251+t576
		t643 = t178*t252*(t285-t414)
		t644 = t254*t583
		t645 = t254*t584
		t651 = -rw*t4*(t493-(t4*t60*t73*t112*t116)/2.0)
		t655 = t6*t9*t115*t603
		t663 = Idyy+Idzz+t24+t135+t306+t345+t394
		t664 = t294+t384+t475
		t671 = rf*t293*t583
		t677 = (t4*t6*t349*t544)/2.0
		t678 = (t6*t9*t349*t544)/2.0
		t681 = t481+t496
		t691 = t6*t662*2.0
		t693 = t480+t509
		t702 = rf*t293*t619
		t705 = t9*t116*t669
		t723 = t497+t540
		t727 = t4*t116*t688
		t729 = -rw*t4*(Idxx*t72*(t285-t414)+t9*t55*t73*t76*t349)
		t736 = t274+t662
		t738 = t72*t76*(Idxx*t72*(t285-t414)+t9*t55*t73*t76*t349)
		t740 = (t493-(t4*t60*t73*t112*t116)/2.0)*(t122-t208)
		t758 = t6*t73*t76*(Idxx*t72*(t285-t414)+t9*t55*t73*t76*t349)
		t762 = t9*t116*(Idxx*t72*(t285-t414)+t9*t55*t73*t76*t349)
		t768 = t212+t270+t378+t385+t402
		t770 = t360+t457+t521
		t776 = t516+t570
		t784 = t554+t561
		t785 = t537+t571
		t786 = t360+t457+t545
		t902 = (Idxx*t6*t73*t869)/4.0
		t903 = (t9*t116*t869)/4.0
		t911 = (t4*t71*t116*t869)/4.0
		t916 = (t221*t869)/4.0
		t917 = t68*t904
		t921 = (rf*t252*t869)/4.0
		t931 = (rf*t293*t869)/4.0
		t948 = (t178*t252*t869)/4.0
		t952 = rf*t9*t116*t293*t869*(-1.0/4.0)
		t953 = (t178*t293*t869)/4.0
		t955 = (t409*t869)/8.0
		t971 = t21+t25+t31+t77+t78+t82+t84+t85+t87+t135+t215+t218+t249+t274+t303+t339+t367
		t1029 = t869*(t285-t414)*(-1.0/4.0)
		t1062 = (t544*t869)/8.0
		t1161 = t99+t101+t103+t104+t105+t106+t107+t172+t304+t323+t358+t404+t407+t479
		t1162 = t99+t101+t103+t104+t105+t106+t107+t172+t304+t323+t358+t404+t407+t476
		t447 = t426*v_ph
		t513 = -t500
		t533 = -t514
		t538 = rw*t9*t507
		t543 = -t532
		t547 = Idxx*t72*t507
		t556 = t4*t116*t507
		t567 = t4*t6*t115*t507
		t573 = t254*t507
		t594 = t217+t321+t439
		t595 = -t591
		t597 = t334*t507
		t598 = t338*t507
		t605 = -t596
		t607 = t320+t524
		t609 = t338+t515
		t611 = t604*2.0
		t617 = Iwyy*t610
		t620 = Iwyy*t615
		t621 = -t618
		t625 = t71*t610
		t626 = t338*t549
		t629 = t6*t9*t349*t507
		t631 = t71*t615
		t632 = t178*t293*t507
		t641 = t71*t624
		t646 = (t409*t507)/2.0
		t647 = t71*t630
		t648 = rw*t4*t630
		t654 = -t645
		t658 = t11*t642
		t659 = t71*t637
		t660 = rw*t4*t637
		t675 = t663*v_om
		t676 = t664*v_om
		t682 = t6*t9*t115*t624
		t683 = t446+t549
		t687 = t460+t536
		t690 = t6*t9*t115*t630
		t698 = t6*t9*t115*t637
		t701 = t71*t681
		t707 = v_th*(t392+v_ph*(t375-t415))
		t708 = t71*t693
		t710 = rw*t4*t693
		t712 = t275+t642
		t713 = rf*t293*t630
		t716 = -t702
		t720 = t336*t624
		t724 = t467+t563
		t726 = rf*t293*t637
		t733 = -t44*(t356-t604)
		t734 = -t727
		t735 = -t630*(t122-t208)
		t737 = t6*t9*t115*t681
		t748 = t448+t589
		t753 = t6*t736
		t761 = (t507*t544)/2.0
		t773 = t336*t681
		t779 = t171+t729
		t787 = -t693*(t122-t208)
		t788 = Tomega+t269+t361+t445+t468
		t793 = rf*t252*t723
		t795 = Idxx*t72*t770
		t797 = Iwyy*t786
		t802 = v_ne*(t368-t405+v_ph*(t375-t415))*-4.0
		t805 = t4*t116*t776
		t806 = t71*t786
		t809 = v_ph*v_th*(t370-t691)
		t813 = Idxx*t72*t786
		t815 = t9*t116*t785
		t816 = t4*t6*t115*t776
		t817 = t6*t9*t115*t776
		t818 = t6*t9*t115*t770
		t830 = -t71*(t532-t588)
		t833 = -rw*t4*(t532-t588)
		t835 = t4*t6*t115*t785
		t836 = t6*t9*t115*t785
		t837 = t4*t6*t115*t784
		t838 = t6*t9*t115*t784
		t842 = -Idxx*t72*(t532-t588)
		t845 = t71*(t532-t588)
		t849 = t254*t784
		t854 = rf*t293*t770
		t859 = t334*t776
		t861 = rf*t252*t784
		t868 = t336*t776
		t871 = rf*t293*t784
		t876 = t334*t785
		t877 = t334*t784
		t879 = t336*t784
		t881 = t336*t785
		t882 = t6*t9*t115*(t532-t588)
		t884 = -t784*(t122-t208)
		t895 = -rf*t293*(t532-t588)
		t897 = t785*(t122-t208)
		t898 = rf*t293*(t532-t588)
		t906 = -t902
		t912 = t71*t903
		t979 = t644+t705
		t989 = t198+t396+t601+t602
		t992 = t374+t917
		t996 = t68*t971
		t1013 = -rw*t4*(t578+t6*t8*t178*(Idxx*t72*(t285-t414)+t9*t55*t73*t76*t349))
		t1014 = Iwyy*(t578+t6*t8*t178*(Idxx*t72*(t285-t414)+t9*t55*t73*t76*t349))
		t1019 = (t507*t869)/4.0
		t1030 = t6*t8*t178*(t580+t11*t115*(t454-t535))
		t1042 = t442+t921
		t1051 = t9*t116*(t578+t6*t8*t178*(Idxx*t72*(t285-t414)+t9*t55*t73*t76*t349))
		t1059 = (t549*t869)/4.0
		t1063 = t452+t931
		t1075 = (t557*t869)/4.0
		t1091 = -rf*t293*(t578+t6*t8*t178*(Idxx*t72*(t285-t414)+t9*t55*t73*t76*t349))
		t1093 = t425+t955
		t1148 = t4*t116*(t412-t955)
		t1150 = t9*t116*(t412-t955)
		t1167 = t528+t948
		t1169 = Iwyy*t1162
		t1176 = t71*t1162
		t1179 = t529+t953
		t1199 = (t122-t208)*(t412-t955)
		t1200 = t336*(t412-t955)
		t1203 = t6*t73*t76*t1161
		t1208 = t254*t1161
		t1209 = t254*t1162
		t1223 = rf*t293*t1161
		t1237 = -t1161*(t122-t208)
		t559 = t71+t538
		t564 = t124+t268+t447
		t613 = t6*t594
		t616 = -t611
		t652 = t72*t99*t607
		t661 = t6*t9*t115*t609
		t665 = (t4*t116*t607)/2.0
		t667 = (t9*t116*t607)/2.0
		t679 = (t4*t6*t115*t607)/2.0
		t680 = (t6*t9*t115*t607)/2.0
		t699 = (t60*t73*t112*t607)/4.0
		t704 = t473+t547
		t709 = -t701
		t717 = (t334*t607)/2.0
		t728 = t6*t712
		t732 = t11*t115*t687
		t739 = (t178*t252*t607)/2.0
		t745 = (t4*t6*t349*t607)/2.0
		t747 = (t6*t9*t349*t607)/2.0
		t751 = (t178*t293*t607)/2.0
		t757 = t284+t676
		t765 = t353+t658
		t774 = t350+t675
		t775 = rf*t55*t63*t178*t683
		t777 = t417+t648
		t782 = t413+t660
		t800 = t556+t569
		t801 = t533+t591
		t810 = rf*t252*t748
		t811 = t543+t588
		t820 = t430+t698
		t827 = -Idxx*t72*(t514+t595)
		t831 = t336*t748
		t832 = t71*(t514+t595)
		t851 = (t544*t607)/4.0
		t863 = -t849
		t864 = -t4*t6*t115*(t514+t595)
		t865 = -t6*t9*t115*(t514+t595)
		t866 = -v_om*(t352-t753)
		t886 = -rf*t293*(t514+t595)
		t887 = rf*t252*(t514+t595)
		t918 = t531+t737
		t920 = t629+t638
		t925 = t617+t651
		t932 = t632+t643
		t951 = t455+t842
		t963 = t624+t713
		t965 = t620+t710
		t1001 = t992*v_ph
		t1004 = Iwyy*t989
		t1005 = rw*t4*t979
		t1007 = t645+t734
		t1024 = t484+t490+t726
		t1026 = Idxx*t72*t989
		t1038 = t4*t116*t989
		t1039 = t9*t116*t989
		t1064 = t622+t795
		t1070 = t254*t989
		t1088 = t623+t813
		t1095 = t9*t116*t1042
		t1097 = t338*t989
		t1104 = t4*t116*t1063
		t1112 = t374+t996
		t1121 = t178*t252*t989
		t1127 = t4*t6*t349*t989
		t1128 = t6*t9*t349*t989
		t1134 = t178*t293*t989
		t1142 = (t607*t869)/8.0
		t1154 = (t409*t989)/2.0
		t1168 = t724+t793
		t1184 = t71*(t882+(t60*t73*t112*t557)/2.0)
		t1185 = rw*t9*(t882+(t60*t73*t112*t557)/2.0)
		t1188 = Iwyy*(t818+t583*(t122-t208))
		t1198 = t4*t116*t1167
		t1205 = t9*t116*t1179
		t1214 = t254*t1167
		t1218 = rf*t252*t1167
		t1222 = t254*t1179
		t1225 = rf*t293*t1167
		t1230 = rf*t252*t1179
		t1231 = t690+t903
		t1232 = t334*t1167
		t1234 = t336*t1167
		t1236 = rf*t293*t1179
		t1239 = t334*t1179
		t1246 = -t1179*(t122-t208)
		t1247 = t797+t833
		t1248 = t776+t871
		t1251 = t785+t861
		t1273 = t682+t952
		t1350 = (t869*t989)/4.0
		t1402 = t838+t1167
		t1409 = t837+t1179
		t1451 = t916+t1169
		t1483 = t548+t621+t787+t898
		t1540 = t35+t302+t605+t697+t733+t809
		t612 = t60*t112*t564
		t627 = t613/2.0
		t672 = -t667
		t714 = t347+t613
		t718 = Iwyy*t704
		t722 = -t717
		t741 = Idxx*t72*t704
		t742 = -t739
		t746 = t728/4.0
		t750 = t4*t116*t704
		t759 = t72*t76*t704
		t767 = t757*v_th*2.0
		t772 = t399+t652
		t778 = t45*t765
		t780 = t6*t8*t178*t704
		t789 = t6*t774*2.0
		t808 = t355+t372+t616
		t819 = rw*t4*t800
		t823 = t4*t6*t115*t777
		t829 = t4*t6*t115*t782
		t843 = rw*t9*t820
		t847 = rf*t252*t777
		t855 = rf*t252*t782
		t857 = -t851
		t862 = t6*t9*t115*t800
		t874 = t336*t777
		t875 = t336*t782
		t885 = rf*t293*t800
		t890 = t4*t116*(t357-t728)*(-1.0/4.0)
		t892 = (Idxx*t6*t73*(t357-t728))/4.0
		t893 = (t9*t116*(t357-t728))/4.0
		t894 = rf*t252*t820

		t900 = t254*(t357-t728)*(-1.0/4.0)
		t905 = t336*t820
		t908 = t60*t73*t112*(t357-t728)*(-1.0/8.0)
		t929 = t330+t866
		t934 = rw*t9*t918
		t935 = Iwyy*t920
		t936 = (t178*t252*(t357-t728))/4.0
		t937 = rw*t4*(t665+t254*(t285-t414))
		t938 = t178*t293*(t357-t728)*(-1.0/4.0)
		t939 = Iwyy*t932
		t942 = rw*t4*t920
		t943 = t409*(t357-t728)*(-1.0/8.0)
		t944 = rw*t4*t932
		t946 = t456+t827
		t956 = t6*t9*t115*t920
		t961 = t4*t6*t115*t925
		t962 = t6*t9*t115*t932
		t964 = t254*t920
		t970 = t598+t745
		t974 = t254*t932
		t976 = rf*t293*t920
		t977 = rw*t9*t963
		t980 = t614+t747
		t981 = t334*t920
		t983 = (t665+t254*(t285-t414))*(t122-t208)
		t986 = rf*t293*t932
		t990 = t336*t925
		t994 = t334*t932
		t995 = (t507*(t357-t728))/4.0
		t998 = -t920*(t122-t208)
		t999 = ((t285-t414)*(t357-t728))/4.0
		t1012 = t4*t6*t115*t965
		t1015 = t4*t6*t115*t963
		t1016 = -t1004
		t1033 = t71*t1007
		t1034 = rw*t9*t1007
		t1036 = t551+t865
		t1037 = (t409*t920)/2.0
		t1046 = rw*t9*t1024
		t1053 = -t1039
		t1065 = t336*t965
		t1066 = t336*t963
		t1069 = -rw*t4*(t751+(t409*(t285-t414))/2.0)
		t1087 = t4*t6*t115*t1024
		t1103 = t607*(t357-t728)*(-1.0/8.0)
		t1111 = t6*t9*t115*(t751+(t409*(t285-t414))/2.0)
		t1126 = t336*t1024
		t1131 = t626+t864
		t1135 = -t1121
		t1139 = t44*t1112
		t1145 = -t334*(t751+(t409*(t285-t414))/2.0)
		t1160 = (t544*t920)/2.0
		t1166 = (t751+(t409*(t285-t414))/2.0)*(t122-t208)
		t1193 = -t1184
		t1194 = -t1185
		t1206 = rw*t9*(t801+t637*(t122-t208))
		t1213 = -t1205
		t1242 = rw*t9*t1231
		t1243 = -t1239
		t1253 = Iwyy*t1248
		t1258 = Iwyy*t1251
		t1261 = rw*t9*t1248
		t1263 = rw*t4*t1251
		t1269 = rf*t252*t1231
		t1277 = t4*t6*t115*t1247
		t1278 = t4*t6*t115*t1248
		t1279 = t336*t1231
		t1285 = rw*t9*t1273
		t1293 = rf*t252*t1247
		t1310 = t336*t1248
		t1321 = t6*t9*t115*(t1038+t334*(t285-t414))
		t1325 = rf*t293*(t1038+t334*(t285-t414))
		t1364 = -Iwyy*(t1134+(t544*(t285-t414))/2.0)
		t1366 = -rw*t4*(t1134+(t544*(t285-t414))/2.0)
		t1371 = Iwyy*(t661-t1223)
		t1386 = t6*t9*t115*(t1134+(t544*(t285-t414))/2.0)
		t1404 = rf*t293*(t1134+(t544*(t285-t414))/2.0)
		t1416 = Idxx*t6*t73*t1402
		t1422 = -t9*t116*(t851-t1154)
		t1431 = t6*t73*t76*t1409
		t1433 = rf*t252*t1402
		t1440 = t336*t1402
		t1459 = -Idxx*t6*t73*(t835-t1230)
		t1461 = t805+t815+t863
		t1466 = t548+t621+t773+t887
		t1480 = rf*t252*t1451
		t1490 = t336*t1451
		t1499 = t71*t1483
		t1500 = rw*t9*t1483
		t1555 = -t4*t6*t115*(-t597+t1039+t800*(t122-t208))
		t1567 = rf*t252*(-t597+t1039+t800*(t122-t208))
		t1573 = t1095+t1208
		t1576 = t1104+t1209
		t1591 = t1150+t1214
		t1595 = t1148+t1222
		t1636 = -rw*t9*(-t761+t1121+t932*(t122-t208))
		t1649 = rf*t252*(-t761+t1121+t932*(t122-t208))
		t1660 = t816+t1093+t1218
		t1667 = t836+t1093+t1236
		t1699 = t4*t6*t115*(t400+t1004+t619*(t122-t208)+rw*t4*(t1038+t334*(t285-t414)))
		t1704 = -rf*t252*(t400+t1004+t619*(t122-t208)+rw*t4*(t1038+t334*(t285-t414)))
		t752 = (rw*t4*t714)/2.0
		t755 = (rw*t9*t714)/2.0
		t763 = t72*t99*t714
		t769 = (t9*t116*t714)/2.0
		t794 = (t60*t73*t112*t714)/4.0
		t796 = (rf*t252*t714)/2.0
		t803 = t333+t718
		t807 = (rf*t293*t714)/2.0
		t814 = t808*v_om*v_ph
		t834 = t6*t8*t178*t772
		t839 = (t178*t252*t714)/2.0
		t840 = (t4*t6*t349*t714)/2.0
		t841 = (t6*t9*t349*t714)/2.0
		t850 = (t409*t714)/4.0
		t852 = (t178*t293*t714)/2.0
		t858 = -t847
		t909 = t406+t513+t612
		t910 = t573+t672
		t923 = (t549*t714)/2.0
		t924 = (t544*t714)/4.0
		t933 = (t557*t714)/2.0
		t941 = t929*v_ph*4.0
		t945 = t379+t567+t627
		t960 = Iwyy*(t379+t627+t6*t9*t115*(t285-t414))
		t972 = -t964
		t978 = Iwyy*t970
		t984 = t579+t780
		t988 = Iwyy*t980
		t1006 = rw*t4*t980
		t1009 = t254*(t379+t627+t6*t9*t115*(t285-t414))
		t1010 = t4*t116*t970
		t1011 = t408+t530+t746
		t1022 = t646+t742
		t1027 = t9*t116*t980
		t1028 = t4*t6*t115*t970
		t1041 = t526+t890
		t1043 = t585+t819
		t1044 = -t1037
		t1048 = -t1033
		t1049 = t6*t9*t115*t980
		t1050 = -t1034
		t1057 = t71*(t408+t746+t6*t9*t349*(t122-t208))
		t1058 = -t1046
		t1060 = t437+t942
		t1061 = rw*t9*t1036
		t1072 = t334*t970
		t1074 = t338*t970
		t1076 = t336*t970
		t1078 = -t1065
		t1082 = -t71*(t525+t893)
		t1089 = t334*t980
		t1096 = t71*(t525+t893)
		t1106 = t178*t293*t970
		t1108 = t980*(t122-t208)
		t1110 = t178*t252*t980
		t1124 = t4*t6*t115*(t525+t893)
		t1138 = -rf*t252*(t525+t893)
		t1140 = -t4*t6*t115*(t431-t944)
		t1155 = t750+t762
		t1156 = rw*t4*t1131
		t1173 = rf*t252*(t431-t944)
		t1187 = t336*(t431-t944)
		t1211 = (t714*(t357-t728))/8.0
		t1212 = rf*t293*t1131
		t1215 = (t544*t970)/2.0
		t1229 = (t544*t980)/2.0
		t1235 = t628+t943
		t1245 = t659+t939
		t1256 = t678+t938
		t1282 = -rf*t252*(t647-t935)
		t1284 = t597+t1053
		t1289 = t4*t6*t115*(t677+t936)
		t1296 = -rf*t252*(t677+t936)
		t1301 = -t1293
		t1306 = (t122-t208)*(t647-t935)
		t1316 = t631+t1069
		t1328 = t525+t735+t893
		t1334 = t789+t1001
		t1348 = t761+t1135
		t1373 = t892+t1026
		t1405 = t908+t1062
		t1407 = t857+t1154
		t1411 = t970+t976
		t1448 = t109+t140+t164+t165+t166+t219+t244+t310+t325+t716+t937
		t1452 = -Idxx*t6*t73*(-t748+t829+t843)
		t1467 = rw*t4*t1461
		t1468 = rw*t9*t1461
		t1474 = t71*t1466
		t1475 = rw*t4*t1466
		t1476 = t677+t884+t936
		t1486 = t999+t1128
		t1487 = t4*t6*t115*t1461
		t1488 = t6*t9*t115*t1461
		t1492 = -Iwyy*(t995+t1127)
		t1498 = t6*t9*t115*t1466
		t1501 = Iwyy*(t995+t1127)
		t1504 = -t1499
		t1505 = -t1500
		t1512 = t336*t1461
		t1529 = t1097+t1103
		t1538 = -rf*t252*(t995+t1127)
		t1580 = rw*t4*t1573
		t1584 = t71*t1576
		t1585 = rw*t9*t1576
		t1594 = t806+t1366
		t1600 = t845+t1364
		t1605 = -t6*t73*t76*t1595
		t1606 = t1198+t1213
		t1616 = t336*t1595
		t1647 = -Idxx*t6*t73*(t1261-t1263)
		t1670 = t995+t998+t1127
		t1672 = t6*t73*t76*t1660
		t1681 = t6*t73*t76*t1667
		t1682 = t334*t1660
		t1691 = t334*t1667
		t1694 = t336*t1667
		t1715 = t875+t1206+t1247
		t1716 = t831+t1194+t1277
		t1740 = t654+t727+t798+t894+t1087
		t1789 = Idxx*t6*t73*(t934+t1005+Iwyy*(t655-t671)+t6*t8*t178*(t581-t732))
		t1790 = t823+t1242+t1451
		t1802 = t722+t983+t1070+t1325
		t1854 = t1015+t1269+t1576
		t1897 = -rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154))
		t1928 = -Idxx*t72*(-t548+t618-t775+t895+t1126+t693*(t122-t208)+rf*t252*(t801+t637*(t122-t208)))
		t2038 = t1278+t1433+t1667
		t812 = t280+t752
		t824 = -t814
		t846 = -t840
		t872 = -t9*t116*(t278-t755)
		t919 = rw*t9*t910
		t922 = (t909*v_ne)/2.0
		t949 = Iwyy*t945
		t969 = t336*t910
		t985 = t254*t945
		t991 = Iwyy*t984
		t1002 = rw*t9*t984
		t1018 = -t1006
		t1025 = t4*t116*t984
		t1031 = Iwyy*t1022
		t1040 = Idxx*t72*t1011
		t1055 = t254*t984
		t1067 = t4*t116*t1022
		t1071 = t71*t1041
		t1073 = rw*t4*t1041
		t1077 = t6*t9*t115*t1011
		t1081 = t599+t834
		t1085 = t598+t839
		t1086 = t4*t6*t115*t1022
		t1100 = t608+t852
		t1101 = rf*t293*t1011
		t1105 = t6*t9*t115*t1041
		t1107 = t4*t6*t115*t1043
		t1116 = t334*t1022
		t1118 = t4*t6*t115*t1060
		t1123 = rf*t252*t1043
		t1125 = t336*t1022
		t1129 = t680+t807
		t1137 = rf*t293*t1041
		t1147 = rf*t252*t1060
		t1151 = t336*t1043
		t1171 = t336*t1060
		t1174 = -t9*t116*(t679-t796)
		t1192 = t72*t76*t1155
		t1195 = t699+t850
		t1201 = t6*t73*t76*t1155
		t1244 = t769+t862
		t1254 = t4*t116*t1235
		t1255 = t9*t116*t1235
		t1264 = t4*t6*t115*t1235
		t1265 = t6*t9*t115*t1235
		t1270 = t763+t906
		t1274 = t9*t116*t1256
		t1275 = t4*t6*t115*t1245
		t1276 = t6*t9*t115*t1245
		t1286 = t6*t9*t115*t1256
		t1288 = rf*t252*t1245
		t1294 = t254*t1256
		t1297 = rf*t293*t1245
		t1299 = rw*t9*t1284
		t1302 = rf*t252*t1256
		t1305 = t336*t1245
		t1311 = rf*t293*t1256
		t1314 = -t1245*(t122-t208)
		t1317 = t4*t6*t115*t1284
		t1320 = rf*t252*t1284
		t1326 = t885+t910
		t1330 = t336*t1316
		t1332 = rw*t9*t1328
		t1336 = t1334*v_th
		t1341 = t4*t6*t115*t1328
		t1343 = rf*t252*t1328
		t1351 = t841+t1029
		t1352 = Iwyy*t1348
		t1355 = -Idxx*(t840-t1019)
		t1370 = t4*t116*t1348
		t1375 = t794+t1142
		t1377 = t4*t6*t115*t1348
		t1380 = t254*t1348
		t1383 = -rf*t252*(t840-t1019)
		t1384 = rf*t252*t1348
		t1385 = Idxx*t72*t1373
		t1387 = rf*t293*t1348
		t1397 = -t336*(t840-t1019)
		t1410 = t178*t293*(t840-t1019)
		t1413 = (t409*(t840-t1019))/2.0
		t1417 = t4*t116*t1405
		t1418 = t9*t116*t1405
		t1419 = rw*t9*t1411
		t1428 = t924+t1097
		t1434 = rf*t252*t1405
		t1437 = t986+t1022
		t1439 = rf*t293*t1405
		t1470 = t4*t6*t115*t1448
		t1473 = t879+t1256
		t1489 = t336*t1448
		t1494 = Iwyy*t1486
		t1495 = -t1488
		t1497 = Idxx*t72*t1476
		t1502 = rw*t4*t1486
		t1509 = t9*t116*t1486
		t1510 = t4*t6*t115*t1476
		t1515 = t1059+t1124
		t1520 = t6*t9*t115*t1486
		t1524 = rf*t252*t1476
		t1531 = t254*t1486
		t1543 = rf*t293*t1486
		t1548 = t4*t116*t1529
		t1549 = t9*t116*t1529
		t1551 = t178*t252*t1486
		t1553 = t4*t6*t115*t1529
		t1554 = t6*t9*t115*t1529
		t1563 = (t409*t1486)/2.0
		t1569 = t338*t1529
		t1575 = t178*t252*t1529
		t1577 = t178*t293*t1529
		t1583 = t933+t1321
		t1588 = -t1585
		t1609 = t858+t925+t977
		t1610 = rf*t252*t1594
		t1615 = t6*t73*t76*t1606
		t1617 = rf*t252*t1600
		t1621 = rf*t252*t1606
		t1623 = rf*t293*t1606
		t1624 = t336*t1606
		t1630 = t855+t965+t1058
		t1631 = t485+t720+t900+t1138
		t1651 = t972+t1010+t1027
		t1656 = -Idxx*(t1211+t1350)
		t1662 = -t4*t116*(t1211+t1350)
		t1674 = rw*t9*t1670

		t1684 = -t178*t293*(t1211+t1350)
		t1685 = -t1682
		t1687 = t409*(t1211+t1350)*(-1.0/2.0)
		t1698 = -t1694
		t1700 = t1096+t1492
		t1709 = t1044+t1106+t1110
		t1725 = Idxx*t72*t1715
		t1732 = t1253+t1467
		t1733 = t1061+t1156+t1188
		t1734 = t1258+t1468
		t1735 = t868+t1235+t1296
		t1736 = t11*t349*t1715
		t1737 = t11*t349*t1716
		t1748 = Idxx*t6*t73*t1740
		t1754 = t393+t422+t767+t802+t941+t1139
		t1783 = t810+t1012+t1030+t1050
		t1798 = t6*t8*t178*t1790
		t1806 = rw*t9*t1802
		t1815 = -t71*(t1212-t1498+t979*(t122-t208))
		t1823 = t1234+t1289+t1405
		t1834 = t1078+t1293+t1500
		t1864 = t6*t73*t76*t1854
		t1889 = t1166+t1404+t1407
		t1903 = rf*t293*(t334*(t840-t1019)+t9*t116*(t1211+t1350))
		t1956 = t1076+t1529+t1538
		t1966 = t1285+t1371+t1580
		t1967 = t959+t1065+t1301+t1505
		t1975 = -Idxx*t72*(t967+t1475+Iwyy*(t854+t603*(t122-t208))+rw*t9*(t886+t681*(t122-t208)))
		t2036 = t1187+t1594+t1636
		t2039 = Iwyy*t2038
		t2040 = t71*t2038
		t2101 = -rf*t252*(t338*(t995+t1127)+(t544*(t840-t1019))/2.0+t178*t252*(t1211+t1350))
		t860 = t4*t116*t812
		t926 = -t922
		t928 = -t919
		t1094 = Iwyy*t1085
		t1109 = Iwyy*t1100
		t1115 = t4*t116*t1085
		t1117 = t4*t116*t1081
		t1119 = t9*t116*t1081
		t1120 = rw*t4*t1100
		t1130 = -t1116
		t1146 = t9*t116*t1100
		t1153 = t254*t1085
		t1159 = -t1147
		t1163 = rf*t252*t1085
		t1172 = t254*t1100
		t1177 = t334*t1085
		t1178 = t4*t116*t1129
		t1183 = t336*t1085
		t1189 = rf*t293*t1100
		t1196 = t334*t1100
		t1204 = -t1100*(t122-t208)
		t1219 = t4*t116*t1195
		t1220 = t9*t116*t1195
		t1249 = t336*t1195
		t1257 = rw*t9*t1244
		t1283 = rf*t252*t1244
		t1303 = t336*t1244
		t1304 = -t1294
		t1309 = t6*t73*t76*t1270
		t1312 = -t1302
		t1324 = t709+t1031
		t1329 = rw*t9*t1326
		t1333 = -t1330
		t1337 = t872+t949
		t1338 = t4*t6*t115*t1326
		t1340 = Iwyy*(t1101-t609*(t122-t208))
		t1342 = (t122-t208)*(t701-t1031)
		t1344 = t336*t1326
		t1345 = -t1343
		t1349 = t846+t1019
		t1358 = Iwyy*t1351
		t1363 = rw*t4*t1351
		t1372 = t9*t116*t1351
		t1381 = t254*t1351
		t1390 = rf*t293*t1351
		t1393 = t334*t1351
		t1394 = t4*t116*t1375
		t1395 = t9*t116*t1375
		t1403 = -t1351*(t122-t208)
		t1406 = t178*t252*t1351
		t1414 = (t409*t1351)/2.0
		t1415 = t336*t1375
		t1424 = -t1417
		t1425 = -t1418
		t1427 = t178*t252*t1375
		t1432 = t178*t293*t1375
		t1441 = -t1439
		t1442 = t4*t116*t1428
		t1443 = t9*t116*t1428
		t1444 = rw*t9*t1437
		t1447 = t962+t1085
		t1455 = rf*t252*t1428
		t1458 = (t544*t1351)/2.0
		t1460 = rf*t293*t1428
		t1463 = t1025+t1051
		t1477 = t336*t1437
		t1482 = (t544*t1375)/2.0
		t1491 = Idxx*t72*t1473
		t1496 = t397+t1002+t1013
		t1503 = t985+t1174
		t1511 = t1075+t1105
		t1533 = rw*t4*t1515
		t1534 = -t1524
		t1539 = -t1531
		t1550 = t1040+t1203
		t1559 = rf*t293*t1515
		t1565 = -t1563
		t1566 = t759+t1355
		t1574 = t923+t1317
		t1581 = t1077+t1237
		t1590 = rw*t9*t1583
		t1593 = t832+t1352
		t1599 = (t122-t208)*(t738-Idxx*t1351)
		t1618 = Idxx*t6*t73*t1609
		t1625 = t6*t8*t178*t1609
		t1634 = t485+t740+t900+t1137
		t1635 = rw*t4*t1631
		t1642 = t6*t9*t115*t1631
		t1643 = t27*t55*t110*t1630
		t1644 = t6*t27*t72*t73*t1630
		t1650 = t11*t349*t1630
		t1653 = t550+t1630
		t1654 = rw*t4*t1651
		t1655 = rw*t9*t1651
		t1661 = t6*t9*t115*t1651
		t1679 = -t1674
		t1683 = rw*t4*(-t974+t1067+t9*t116*(t751+(t409*(t285-t414))/2.0))
		t1689 = -t6*t9*t115*(-t974+t1067+t9*t116*(t751+(t409*(t285-t414))/2.0))
		t1690 = -rf*t293*(t373-t689+t1016+t1299)
		t1701 = -t336*(-t974+t1067+t9*t116*(t751+(t409*(t285-t414))/2.0))
		t1706 = (t122-t208)*(-t974+t1067+t9*t116*(t751+(t409*(t285-t414))/2.0))
		t1718 = t6*t9*t115*t1709
		t1719 = t4*t6*t115*t1709
		t1728 = t334*t1709
		t1729 = t336*t1709
		t1742 = Idxx*t72*t1735
		t1743 = t6*t73*t76*t1732
		t1745 = t897+t1235+t1311
		t1746 = t6*t9*t115*t1735
		t1758 = t11*t349*t1733
		t1762 = -Idxx*t72*(-t877+t1274+t4*t116*(t677+t936))
		t1772 = t4*t6*t115*(-t877+t1274+t4*t116*(t677+t936))
		t1773 = -rf*t252*(-t877+t1274+t4*t116*(t677+t936))
		t1785 = t6*t73*t76*t1783
		t1793 = t1416+t1497
		t1818 = t1200+t1264+t1434
		t1819 = t6*t8*t178*(t874+t1073-t1332+Iwyy*(t408+t746+t6*t9*t349*(t122-t208)))
		t1827 = rw*t4*(-t994+t1370+t9*t116*(t1134+(t544*(t285-t414))/2.0))
		t1829 = t1246+t1286+t1405
		t1831 = t1028+t1375+t1383
		t1839 = t254*t1823
		t1842 = rf*t293*(-t994+t1370+t9*t116*(t1134+(t544*(t285-t414))/2.0))
		t1844 = rf*t293*t1823
		t1859 = t11*t349*t1834
		t1875 = -rw*t4*(t851-t1125-t1154+t1384)
		t1893 = Iwyy*t1889
		t1895 = rw*t9*t1889
		t1896 = -t6*t9*t115*(t1116-t1380+t9*t116*(t851-t1154))
		t1935 = -Idxx*(-t981+t1509+t4*t116*(t995+t1127))
		t1938 = -rw*t9*(-t981+t1509+t4*t116*(t995+t1127))
		t1939 = rw*t4*(-t981+t1509+t4*t116*(t995+t1127))
		t1942 = t1452+t1725
		t1948 = t961+t1480+t1588
		t1961 = t1108+t1529+t1543
		t1963 = t6*t9*t115*t1956
		t1970 = -t6*t9*t115*(-t1072+t1549+t254*(t995+t1127))
		t1971 = Idxx*t72*t1967
		t1972 = Idxx*t6*t73*t1966
		t1976 = -t4*t6*t115*(-t1160+t1551+t178*t293*(t995+t1127))
		t1980 = -t254*(-t1160+t1551+t178*t293*(t995+t1127))
		t1981 = -rf*t252*(-t1160+t1551+t178*t293*(t995+t1127))
		t1994 = t4*t116*(-t1215+t1575+(t409*(t995+t1127))/2.0)
		t1996 = -t4*t6*t115*(-t1215+t1575+(t409*(t995+t1127))/2.0)
		t1997 = -t6*t9*t115*(-t1215+t1575+(t409*(t995+t1127))/2.0)
		t2032 = rf*t6*t8*(t1798+t11*t349*(-t748+t829+t843))
		t2033 = -t11*t349*(t708+t1288+Iwyy*(t751+(t409*(t285-t414))/2.0)+rw*t9*(-t974+t1067+t9*t116*(t751+(t409*(t285-t414))/2.0)))
		t2037 = Idxx*t72*t2036
		t2042 = t11*t349*t2036
		t2044 = -Idxx*(t400+t1004+t1151+t619*(t122-t208)+rw*t9*(-t597+t1039+t800*(t122-t208))+rw*t4*(t1038+t334*(t285-t414)))
		t2068 = t1495+t1591+t1623
		t2069 = t71*(t1487-t1595+t1621)
		t2070 = rw*t4*(t1487-t1595+t1621)
		t2146 = t6*t73*t76*(Idxx*t72*(t874+t1073-t1332+Iwyy*(t408+t746+t6*t9*t349*(t122-t208)))+t6*t73*t76*t1790)
		t2147 = Idxx*t6*t73*(Idxx*t72*(t874+t1073-t1332+Iwyy*(t408+t746+t6*t9*t349*(t122-t208)))+t6*t73*t76*t1790)
		t2151 = t1748+t1928
		t2170 = rw*t9*(-t859+t1255+t254*(t677+t936)+t1461*(t122-t208)+rf*t293*(-t877+t1274+t4*t116*(t677+t936)))
		t2183 = t1489+t1704+t1806
		t2184 = rw*t9*(-t1232+t1418+t1606*(t122-t208)+t6*t9*t115*(-t877+t1274+t4*t116*(t677+t936)))
		t2197 = t1789+t1975
		t2201 = t6*t73*t76*(Idxx*t72*(t708+t1288+Iwyy*(t751+(t409*(t285-t414))/2.0)+rw*t9*(-t974+t1067+t9*t116*(t751+(t409*(t285-t414))/2.0)))+t6*t73*t76*t1734)
		t2309 = -rw*t9*(-t1072+t1549+t254*(t995+t1127)+t1651*(t122-t208)+rf*t293*(-t981+t1509+t4*t116*(t995+t1127)))
		t2310 = rw*t9*(-t1072+t1549+t254*(t995+t1127)+t1651*(t122-t208)+rf*t293*(-t981+t1509+t4*t116*(t995+t1127)))
		t2333 = -rw*t9*(-t1215+t1575+(t409*(t995+t1127))/2.0+t1709*(t122-t208)+rf*t293*(-t1160+t1551+t178*t293*(t995+t1127)))
		t2336 = -t4*t6*t115*(-t1215+t1575+(t409*(t995+t1127))/2.0+t1709*(t122-t208)+rf*t293*(-t1160+t1551+t178*t293*(t995+t1127)))
		t1132 = -t1119
		t1158 = -t1146
		t1226 = -t1219
		t1227 = -t1220
		t1228 = t450+t1094
		t1240 = t449+t1109
		t1319 = t600+t1120
		t1331 = -t1329
		t1360 = -rf*t252*(t860+t960)
		t1361 = rf*t293*t1337
		t1368 = -t336*(t860+t960)
		t1398 = -t1394
		t1423 = t109+t140+t164+t165+t166+t219+t244+t310+t325+t666+t928
		t1435 = -t1432
		t1449 = -t1442
		t1450 = -t1443
		t1453 = rw*t9*t1447
		t1471 = rw*t4*t1463
		t1472 = rw*t9*t1463
		t1514 = rw*t4*t1503
		t1517 = t1009+t1178
		t1522 = t71*t1511
		t1525 = rw*t9*t1511
		t1528 = t6*t73*t76*t1496
		t1536 = t881+t1312
		t1579 = rw*t4*t1574
		t1586 = Iwyy*t1581
		t1592 = t334*t1566
		t1611 = rf*t293*t1593
		t1628 = t956+t1349
		t1639 = t71*t1634
		t1640 = rw*t9*t1634
		t1702 = t1232+t1425
		t1708 = t1239+t1424
		t1722 = -t1719
		t1738 = t1086+t1163+t1195
		t1747 = t876+t1254+t1304
		t1749 = t1111+t1189+t1195
		t1753 = Idxx*t72*t1745
		t1761 = -Idxx*(t1372+t4*t116*(t840-t1019))
		t1765 = -rw*t9*(t1372+t4*t116*(t840-t1019))
		t1766 = t4*t6*t115*t1745
		t1781 = (t122-t208)*(t1372+t4*t116*(t840-t1019))
		t1786 = (t1395+t254*(t840-t1019))*(t122-t208)
		t1794 = Iwyy*t1793
		t1795 = rw*t9*t1793
		t1796 = t1431+t1491
		t1800 = t860+t960+t1107+t1257
		t1810 = t508+t527+t778+t824+t926+t1336
		t1814 = t625+t1018+t1159+t1419
		t1822 = t1199+t1265+t1441
		t1830 = t9*t116*t1818
		t1836 = t1049+t1375+t1390
		t1837 = t1173+t1316+t1444
		t1845 = t254*t1829
		t1849 = rf*t252*t1829
		t1852 = t334*t1831
		t1866 = t254*(t1406+t1410+t338*t920)
		t1868 = rf*t252*(t1406+t1410+t338*t920)
		t1871 = rf*t293*(t1406+t1410+t338*t920)
		t1873 = t334*(t1406+t1410+t338*t920)
		t1874 = t336*(t1406+t1410+t338*t920)
		t1878 = -(t122-t208)*(t1406+t1410+t338*t920)
		t1879 = t1130+t1380+t1422
		t1881 = t1393+t1662
		t1888 = t1074+t1413+t1427
		t1891 = t1183+t1377+t1428
		t1904 = t1204+t1386+t1428
		t1929 = t1279+t1341+t1511
		t1949 = Idxx*t6*t73*t1942
		t1950 = t4*t116*(t1460+t1195*(t122-t208)+t6*t9*t115*(t851-t1154))
		t1951 = t72*t76*t1942
		t1952 = t6*t73*t76*t1942
		t1954 = t1066+t1345+t1634
		t1955 = Idxx*t6*t73*t1948
		t1957 = -Idxx*t6*t73*(t1625-t1650)
		t1959 = -t11*t115*(t1625-t1650)
		t1962 = Iwyy*t1961
		t1964 = t1089+t1539+t1548
		t1985 = t6*t8*t178*(t988+t1282+t1655-t71*(t493-(t4*t60*t73*t112*t116)/2.0))
		t1986 = t6*t8*t178*(t641-t978-t1654+rf*t293*(t647-t935))
		t1987 = -t336*(t641-t978-t1654+rf*t293*(t647-t935))
		t1991 = t1229+t1565+t1577
		t2009 = t1303+t1555+t1583
		t2012 = t1297+t1324+t1683
		t2014 = t1615+t1762
		t2017 = -Idxx*t72*(-t1340+t1635+rw*t9*(t624*(t122-t208)+rf*t293*(t525+t893)))
		t2025 = -t6*t8*t178*(-t1340+t1635+rw*t9*(t624*(t122-t208)+rf*t293*(t525+t893)))
		t2031 = t1309+t1385+t1656
		t2045 = t1201+t1935
		t2046 = t72*t2044
		t2047 = t1672+t1742
		t2055 = t512+t2044
		t2059 = t1211+t1350+t1403+t1520
		t2071 = rw*t9*t2068
		t2072 = t71*t2068
		t2073 = -t2069
		t2076 = t1310+t1534+t1745
		t2080 = t336*t2068
		t2096 = t9*t116*(t1458+t1684+t338*t1486)
		t2099 = t1057+t1171+t1502+t1679
		t2102 = t254*(t1458+t1684+t338*t1486)
		t2104 = rf*t252*(t1458+t1684+t338*t1486)
		t2106 = rf*t293*(t1458+t1684+t338*t1486)
		t2113 = -t71*(t1559-t1642+t1573*(t122-t208))
		t2115 = t1482+t1569+t1687
		t2118 = t6*t8*t178*(t912+Iwyy*(t840-t1019)+rw*t4*(t1372+t4*t116*(t840-t1019))+t6*t9*t115*(t647-t935))
		t2126 = t1736+t1819
		t2131 = t1344+t1567+t1802
		t2138 = t1314+t1593+t1827
		t2153 = t1440+t1510+t1829
		t2159 = t1333+t1610+t1895
		t2171 = t1785+t1971
		t2178 = t1243+t1417+t1624+t1772
		t2190 = Idxx*t72*t2183
		t2192 = t11*t115*t2183
		t2202 = t1477+t1649+t1889
		t2204 = t1082+t1306+t1501+t1939
		t2231 = -rw*t9*(t1395+t1661+t254*(t840-t1019)+rf*t293*(t1372+t4*t116*(t840-t1019)))
		t2235 = -t336*(t1395+t1661+t254*(t840-t1019)+rf*t293*(t1372+t4*t116*(t840-t1019)))
		t2251 = -Iwyy*(-t1746+t1844+t1660*(t122-t208))
		t2263 = t1504+t1893+t1897
		t2349 = Iwyy*(-t1963+t1831*(t122-t208)+rf*t293*(t1211+t1350+t1397+t4*t6*t115*(t995+t1127)))
		t1445 = t6*t9*t115*t1423
		t1526 = -t1514
		t1537 = rw*t9*t1517
		t1547 = Idxx*t72*t1536
		t1560 = t1115+t1158
		t1598 = t1153+t1227
		t1602 = t1172+t1226

		t1629 = rw*t9*t1628
		t1641 = -t1639
		t1693 = t1177+t1450
		t1703 = t1196+t1449
		t1712 = t465+t991+t1471
		t1713 = t451+t1014+t1472
		t1723 = rf*t293*t1702
		t1724 = rf*t252*t1708
		t1752 = Iwyy*t1749
		t1757 = Idxx*t72*t1747
		t1759 = t334*t1738
		t1768 = t4*t6*t115*t1747
		t1774 = t334*t1749
		t1777 = t1381+t1398
		t1799 = Iwyy*t1796
		t1803 = rw*t4*t1796
		t1805 = rf*t6*t8*t1800
		t1807 = Idxx*t6*t73*t1800
		t1826 = t6*t8*t178*t1814
		t1833 = t4*t116*t1822
		t1840 = Iwyy*t1836
		t1841 = t1140+t1319+t1453
		t1846 = Idxx*t72*t1837
		t1850 = -t1845
		t1855 = t1123+t1331+t1448
		t1857 = t334*t1836
		t1862 = t1192+t1761
		t1865 = t11*t349*t1837
		t1870 = -t1866
		t1886 = rw*t4*t1881
		t1900 = t4*t116*t1888
		t1905 = t254*t1891
		t1906 = t9*t116*(t1414+t1435+(t60*t73*t112*t980)/2.0)
		t1907 = Iwyy*t1904
		t1909 = t334*t1888
		t1910 = t336*t1888
		t1912 = rf*t293*t1891
		t1916 = t334*(t1414+t1435+(t60*t73*t112*t980)/2.0)
		t1917 = t336*(t1414+t1435+(t60*t73*t112*t980)/2.0)
		t1919 = t254*t1904
		t1923 = (t122-t208)*(t1414+t1435+(t60*t73*t112*t980)/2.0)
		t1930 = t1283+t1338+t1517
		t1934 = t6*t8*t178*t1929
		t1958 = Idxx*t72*t1954
		t1965 = t6*t8*t178*t1954
		t1969 = rw*t4*t1964
		t1992 = -Idxx*t72*(t990-t1640+rf*t252*(t1073+Iwyy*(t408+t746+t6*t9*t349*(t122-t208))))
		t1998 = t9*t116*t1991
		t2002 = t4*t6*t115*t1991
		t2003 = t6*t9*t115*t1991
		t2006 = t6*t8*t178*(t990-t1640+rf*t252*(t1073+Iwyy*(t408+t746+t6*t9*t349*(t122-t208))))
		t2013 = rf*t6*t8*t2009
		t2018 = Idxx*t72*t2012
		t2019 = rw*t4*t2014
		t2020 = rw*t9*t2014
		t2024 = t4*t6*t115*t2012
		t2028 = t336*t2012
		t2029 = t11*t349*t2012
		t2034 = t4*t116*t2031
		t2035 = t9*t116*t2031
		t2049 = t6*t8*t178*(-t1490+t1525+t4*t6*t115*(t1073+Iwyy*(t408+t746+t6*t9*t349*(t122-t208))))
		t2050 = t6*t8*t178*t2046
		t2052 = t1681+t1753
		t2054 = -t6*t8*t178*(t1533-t1586+rw*t9*(t903*(t122-t208)+t6*t9*t115*(t525+t893)))
		t2060 = Iwyy*t2059
		t2064 = t254*t2059
		t2074 = -t2071
		t2075 = -t2072
		t2077 = Iwyy*t2076
		t2079 = t71*t2076
		t2089 = t1368+t1590+t1699
		t2105 = -t2102
		t2107 = -t2104
		t2114 = t6*t8*t178*t2099
		t2120 = t4*t116*t2115
		t2121 = t9*t116*t2115
		t2122 = t6*t8*t178*(t911-t1358+t1765+t4*t6*t115*(t647-t935))
		t2127 = Idxx*t2126
		t2134 = t11*t115*t2131
		t2140 = Idxx*t72*t2138
		t2143 = t4*t6*t115*t2138
		t2144 = rf*t252*t2138
		t2148 = t11*t349*t2138
		t2154 = Iwyy*t2153
		t2155 = t71*t2153
		t2164 = t1512+t1747+t1773
		t2172 = -Idxx*t72*(t1690+t1423*(t122-t208)+rw*t4*(t717-t969-t1070+t1320))
		t2185 = t71*t2178
		t2187 = rw*t4*t2178
		t2207 = t4*t6*t115*t2204
		t2209 = t6*t8*t178*t2204
		t2210 = rf*t252*t2204
		t2249 = -t11*t115*(t2025+t11*t349*(t1475+Iwyy*(t854+t603*(t122-t208))+rw*t9*(t886+t681*(t122-t208))))
		t2254 = t1698+t1766+t1849
		t2260 = t1972+t2017
		t2266 = t1985+t2033
		t2270 = t1685+t1830+t1839
		t2275 = t1706+t1842+t1879
		t2283 = t1718+t1871+t1888
		t2288 = Iwyy*(t1414+t1435+t1722+t1868+(t60*t73*t112*t980)/2.0)
		t2289 = rw*t4*(t1414+t1435+t1722+t1868+(t60*t73*t112*t980)/2.0)
		t2303 = -rw*t9*(-t1781+t334*(t840-t1019)+t9*t116*(t1211+t1350)+t6*t9*t115*(-t981+t1509+t4*t116*(t995+t1127)))
		t2320 = t1786+t1903+t1970
		t2330 = t1729+t1981+t1991
		t2352 = Iwyy*(t1458+t1684+t1874+t1976+t338*t1486)
		t2354 = rw*t4*(t1458+t1684+t1874+t1976+t338*t1486)
		t2356 = -rw*t9*(t1878+t338*(t995+t1127)+(t544*(t840-t1019))/2.0+t178*t252*(t1211+t1350)+t6*t9*t115*(-t1160+t1551+t178*t293*(t995+t1127)))
		t2357 = -rf*t252*(t1878+t338*(t995+t1127)+(t544*(t840-t1019))/2.0+t178*t252*(t1211+t1350)+t6*t9*t115*(-t1160+t1551+t178*t293*(t995+t1127)))
		t2365 = -rw*t4*(-t1852+t254*(t1211+t1350+t1397+t4*t6*t115*(t995+t1127))+t9*t116*(-t1415+t1553+rf*t252*(t1211+t1350)))
		t2371 = -rf*t252*(-t1873+t2096+t4*t116*(t338*(t995+t1127)+(t544*(t840-t1019))/2.0+t178*t252*(t1211+t1350)))
		t2444 = -Idxx*(Iwyy*(t1878+t338*(t995+t1127)+(t544*(t840-t1019))/2.0+t178*t252*(t1211+t1350)+t6*t9*t115*(-t1160+t1551+t178*t293*(t995+t1127)))+t71*(-t1232+t1418+t1606*(t122-t208)+t6*t9*t115*(-t877+t1274+t4*t116*(t677+t936)))+rw*t4*(-t1873+t2096+t4*t116*(t338*(t995+t1127)+(t544*(t840-t1019))/2.0+t178*t252*(t1211+t1350))))
		t1570 = rw*t4*t1560
		t1571 = rw*t9*t1560
		t1589 = rf*t293*t1560
		t1607 = rw*t4*t1602
		t1622 = t1598*(t122-t208)
		t1710 = rw*t4*t1703
		t1714 = rf*t293*t1693
		t1720 = t6*t73*t76*t1712
		t1727 = -t1724
		t1755 = -t1752
		t1778 = -t1774
		t1780 = rw*t4*t1777
		t1804 = -t1803
		t1809 = t1459+t1547
		t1843 = -t1840
		t1851 = Idxx*t6*t73*t1841
		t1858 = Idxx*t72*t1855
		t1863 = t11*t115*t1855
		t1867 = t11*t349*t1841
		t1882 = t6*t8*t72*t76*t178*t1855
		t1908 = -t1907
		t1931 = Idxx*t6*t73*t1930
		t1990 = t1361+t1445+t1526
		t2000 = -t1998
		t2005 = t1360+t1470+t1537
		t2007 = t1605+t1757
		t2021 = -t2019
		t2022 = -t2020
		t2053 = t1118+t1176+t1363+t1629
		t2061 = -t2060
		t2093 = rf*t6*t8*t2089
		t2100 = t1647+t1846
		t2130 = t6*t73*t2127
		t2145 = -t2144
		t2152 = rf*t6*t8*(t1934-t11*t349*(t882-t905+t4*t6*t115*(t801+t637*(t122-t208))+(t60*t73*t112*t557)/2.0))
		t2167 = t71*t2164
		t2168 = rw*t4*t2164
		t2188 = rw*t9*(t1723+t1591*(t122-t208)+t6*t9*t115*(-t859+t1255+t254*(t677+t936)))
		t2196 = t1743+t2018
		t2205 = t1737+t2049
		t2211 = -t2209
		t2212 = -t2210
		t2218 = rw*t9*(-t1177+t1443+t1560*(t122-t208)+t6*t9*t115*(-t994+t1370+t9*t116*(t1134+(t544*(t285-t414))/2.0)))
		t2220 = t1864+t1958
		t2224 = t1758+t2054
		t2225 = rf*t252*(-t1177+t1443+t1560*(t122-t208)+t6*t9*t115*(-t994+t1370+t9*t116*(t1134+(t544*(t285-t414))/2.0)))
		t2237 = t1807+t2046
		t2240 = t1859+t2006
		t2252 = t1955+t1992
		t2256 = Iwyy*t2254
		t2261 = Idxx*t6*t73*t2260
		t2265 = t1986+t2029
		t2269 = t11*t115*t2266
		t2271 = rw*t4*t2270
		t2272 = t1691+t1833+t1850
		t2276 = rw*t9*t2275
		t2280 = t4*t6*t115*t2275
		t2284 = Iwyy*t2283
		t2285 = rw*t9*t2283
		t2290 = t336*t2283
		t2300 = t1641+t1962+t1969
		t2311 = t2042+t2114
		t2321 = -t6*t8*t178*(t2013-t2134)
		t2322 = rw*t9*t2320
		t2324 = t1870+t1900+t1906
		t2332 = Iwyy*t2330
		t2334 = rw*t4*t2330
		t2359 = t1917+t2002+t2107
		t2367 = -rw*t9*(-t1857+t2064+t4*t116*(-t1554+t1375*(t122-t208)+rf*t293*(t1211+t1350)))
		t2374 = t1916+t2105+t2120
		t2375 = t2039+t2070+t2074
		t2392 = t1910+t1996+t2101+t2115
		t2393 = t1923+t2003+t2106+t2115
		t2398 = t2154+t2184+t2187
		t2414 = -t71*(-t1691-t1833+t1845+t2080+rf*t252*(-t1232+t1418+t1606*(t122-t208)+t6*t9*t115*(-t877+t1274+t4*t116*(t677+t936)))+t4*t6*t115*(-t859+t1255+t254*(t677+t936)+t1461*(t122-t208)+rf*t293*(-t877+t1274+t4*t116*(t677+t936))))
		t2424 = -t6*t8*t178*(t1639-t1962-t1969+t2210+t2309+t336*(t641-t978-t1654+rf*t293*(t647-t935)))
		t2426 = t2155+t2354+t2356
		t2446 = -Idxx*(-t2185+t2352+rw*t9*(-t1873+t2096+t4*t116*(t338*(t995+t1127)+(t544*(t840-t1019))/2.0+t178*t252*(t1211+t1350))))
		t2452 = -Idxx*(-t1857+t2064+t2235+rf*t252*(-t1781+t334*(t840-t1019)+t9*t116*(t1211+t1350)+t6*t9*t115*(-t981+t1509+t4*t116*(t995+t1127)))+t4*t116*(-t1554+t1375*(t122-t208)+rf*t293*(t1211+t1350))+t4*t6*t115*(-t1072+t1549+t254*(t995+t1127)+t1651*(t122-t208)+rf*t293*(-t981+t1509+t4*t116*(t995+t1127))))
		t1572 = -t1571
		t1608 = -t1607
		t1711 = -t1710
		t1902 = t1228+t1276+t1570
		t1995 = Idxx*t6*t73*t1990
		t2010 = t6*t73*t76*t2005
		t2056 = t6*t8*t178*t2053
		t2062 = t1033+t1607+t1755
		t2088 = t1618+t1858
		t2112 = t6*t73*t76*t2100
		t2125 = t1589+t1598+t1689
		t2135 = t1795+t1804
		t2156 = t1805+t1863
		t2180 = t1184+t1710+t1908
		t2189 = t1616+t1727+t1768
		t2200 = t6*t73*t76*t2196
		t2208 = rf*t6*t8*t2205
		t2213 = t1794+t2021
		t2215 = t1799+t2022
		t2227 = Idxx*t6*t73*t2220
		t2228 = t6*t73*t76*t2220
		t2229 = t1882+t1957
		t2230 = rf*t6*t8*t2224
		t2238 = Idxx*t72*t2237
		t2239 = t1584+t1780+t1843
		t2241 = t1622+t1714+t1896
		t2243 = t11*t115*t2240
		t2245 = t1851+t2037
		t2253 = Idxx*t6*t73*t2252
		t2267 = t11*t115*t2265
		t2273 = rw*t9*t2272
		t2292 = t72*t76*(t1931-Idxx*t72*t2131)
		t2302 = t4*t6*t115*t2300
		t2313 = Idxx*t2311
		t2314 = t1522+t1886+t2061
		t2323 = t2050+t2130
		t2325 = rw*t4*t2324
		t2326 = rw*t9*t2324
		t2327 = t336*t2324
		t2335 = -t2334
		t2345 = -t6*t8*t178*(t2093-t2192)
		t2362 = t2148+t2211
		t2377 = t6*t73*t76*t2375
		t2378 = t1728+t1980+t1994+t2000
		t2388 = t2077+t2168+t2170
		t2399 = Idxx*t2398
		t2401 = t1959+t2032+t2375
		t2412 = -Idxx*(-t2079+t2334+rw*t9*(-t1215+t1575+(t409*(t995+t1127))/2.0+t1709*(t122-t208)+rf*t293*(-t1160+t1551+t178*t293*(t995+t1127))))
		t2417 = t2028+t2145+t2263+t2276
		t2422 = t1987+t2212+t2300+t2310
		t2427 = Idxx*t2426
		t2455 = t2113+t2322+t2349+t2365
		t2471 = t2290+t2336+t2357+t2393
		t1911 = t1240+t1275+t1572
		t1914 = Idxx*t6*t73*t1902
		t1921 = rf*t252*t1902
		t1924 = t336*t1902
		t1925 = t11*t349*t1902
		t2057 = -t2056
		t2078 = t336*t2062
		t2091 = Idxx*t6*t73*t2088
		t2128 = rw*t9*t2125
		t2132 = t336*t2125
		t2136 = Idxx*t6*t73*t2135
		t2157 = t6*t8*t178*t2156
		t2158 = t11*t349*t2156
		t2191 = rw*t4*t2189
		t2194 = rf*t252*t2180
		t2219 = Idxx*t6*t73*t2213
		t2221 = Idxx*t6*t73*t2215
		t2233 = Idxx*t6*t73*t2229
		t2242 = rw*t9*t2241
		t2244 = t336*t2239
		t2248 = t72*t76*t2245
		t2268 = -t2267
		t2317 = rf*t252*t2314
		t2337 = t1995+t2172
		t2339 = t2010+t2190
		t2341 = t1528+t2313
		t2363 = Idxx*t2362
		t2379 = rw*t4*t2378
		t2380 = rw*t9*t2378
		t2383 = t4*t6*t115*t2378
		t2384 = t6*t9*t115*t2378
		t2389 = Idxx*t2388
		t2400 = t6*t73*t2399
		t2403 = t6*t73*t76*t2401
		t2409 = t6*t8*t178*(t2231+t2239+rf*t252*(t912+Iwyy*(t840-t1019)+rw*t4*(t1372+t4*t116*(t840-t1019))+t6*t9*t115*(t647-t935))+t4*t6*t115*(t641-t978-t1654+rf*t293*(t647-t935)))
		t2411 = t2079+t2333+t2335
		t2433 = t76*(-t2207+t2303+t2314+t336*(t912+Iwyy*(t840-t1019)+rw*t4*(t1372+t4*t116*(t840-t1019))+t6*t9*t115*(t647-t935)))
		t2449 = t2188+t2230+t2249+t2251+t2271
		t2456 = -t76*t2455
		t2472 = Iwyy*t2471
		t1927 = t11*t349*t1911
		t2129 = -t2128
		t2257 = t1867+t2057
		t2274 = t1925+t2118
		t2295 = t1914+t2140
		t2299 = t72*t76*(Idxx*t72*(t830+t1305+Iwyy*(t1134+(t544*(t285-t414))/2.0)+rw*t9*(-t994+t1370+t9*t116*(t1134+(t544*(t285-t414))/2.0)))+t6*t73*t76*t1911)
		t2318 = -t2317
		t2338 = Idxx*t72*t2337
		t2382 = -t2380
		t2385 = -t2383
		t2387 = t1720+t2363
		t2390 = t72*t2389
		t2391 = t6*t73*t2389
		t2404 = t1193+t1711+t1907+t1924+t2143+t2218
		t2431 = -Idxx*(t2379+Iwyy*(-t1215+t1575+(t409*(t995+t1127))/2.0+t1709*(t122-t208)+rf*t293*(-t1160+t1551+t178*t293*(t995+t1127)))+t71*(-t859+t1255+t254*(t677+t936)+t1461*(t122-t208)+rf*t293*(-t877+t1274+t4*t116*(t677+t936))))
		t2448 = t2191+t2208+t2243+t2256+t2273
		t2458 = t1778+t1919+t1950+t2132+t2225+t2280+t2321
		t2463 = t2147+t2238+t2433
		t2475 = rw*t9*(-t1909+t2121-t2384+t254*(t338*(t995+t1127)+(t544*(t840-t1019))/2.0+t178*t252*(t1211+t1350))+t2324*(t122-t208)+rf*t293*(-t1873+t2096+t4*t116*(t338*(t995+t1127)+(t544*(t840-t1019))/2.0+t178*t252*(t1211+t1350))))
		t2484 = t11*t115*(t2233+Idxx*(t2424+t11*t349*(t1499-t1893-t2028+t2144-t2276+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154)))))
		t2258 = rf*t6*t8*t2257
		t2278 = rf*t6*t8*t2274
		t2279 = t1927+t2122
		t2297 = t72*t76*t2295
		t2394 = t1048+t1608+t1752+t1921+t2024+t2129
		t2395 = Idxx*t6*t73*(t1921+t2024-t2062+t2129)
		t2397 = t11*t349*(t1921+t2024-t2062+t2129)
		t2405 = Idxx*t2404
		t2429 = t2167+t2332+t2382
		t2434 = t2377+t2390
		t2453 = t2244+t2302+t2318+t2367
		t2464 = t6*t8*t178*t2463
		t2476 = t2327+t2371+t2374+t2385
		t2485 = t2484*2.0
		t2486 = t2484*4.0
		t2488 = t2484*8.0
		t2281 = rf*t6*t8*t2279
		t2406 = t72*t2405
		t2410 = t2157+t2394
		t2415 = t1952+t2405
		t2430 = Idxx*t2429
		t2435 = Idxx*t6*t73*t2434
		t2438 = t6*t73*t76*t2434
		t2459 = t2075+t2268+t2278+t2284+t2325
		t2462 = t2397+t2409
		t2465 = -t2464
		t2467 = t72*t76*(t2395+Idxx*t72*(t1499-t1893-t2028+t2144-t2276+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154))))
		t2468 = Idxx*t72*(t2395+Idxx*t72*(t1499-t1893-t2028+t2144-t2276+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154))))*-2.0
		t2469 = Idxx*t72*(t2395+Idxx*t72*(t1499-t1893-t2028+t2144-t2276+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154))))*-4.0
		t2470 = Idxx*t72*(t2395+Idxx*t72*(t1499-t1893-t2028+t2144-t2276+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154))))*-8.0
		t2477 = rw*t4*t2476
		t2282 = -t2281
		t2416 = t11*t349*t2415
		t2437 = t2435*4.0
		t2439 = t2435*-2.0
		t2441 = t2435*8.0
		t2494 = t2414+t2472+t2475+t2477
		t2440 = -t2437
		t2442 = -t2441
		t2461 = t2073+t2269+t2282+t2288+t2326
		t2489 = t2416+t2465
		t2495 = Idxx*t2494
		t2496 = t18*t2494
		t2490 = rf*t6*t8*t2489
		t2497 = t2495*4.0
		t2498 = t2495*8.0
		t2491 = t2490*2.0
		t2492 = t2490*4.0
		t2493 = t2490*8.0
		t2499 = t2438+t2467+t2484+t2490+t2495
		t2500 = t2439+t2468+t2485+t2491+t2496
		t2501 = t2440+t2469+t2486+t2492+t2497

		t2502 = 1.0/t2499
		t2505 = t2442+t2470+t2488+t2493+t2498
		t2503 = 1.0/t2500
		t2504 = 1.0/t2501
		t2506 = 1.0/t2505

		et1 = t76*(t71*(t1723+t1591*(t122-t208)+t6*t9*t115*(-t859+t1255+t254*(t677+t936)))-Iwyy*(t1997+t1888*(t122-t208)+rf*t293*(t338*(t995+t1127)+(t544*(t840-t1019))/2.0+t178*t252*(t1211+t1350)))+rw*t4*(-t1909+t2121+t254*(t338*(t995+t1127)+(t544*(t840-t1019))/2.0+t178*t252*(t1211+t1350))))-t11*t115*(Idxx*(t11*t349*(-t1342+t1611+rw*t4*(t1116-t1380+t9*t116*(t851-t1154)))+t6*t8*t178*((t122-t208)*(t641-t978)+rw*t4*(-t1072+t1549+t254*(t995+t1127))+rf*t293*t1700))+t6*t73*t76*(rf*t293*(t465+t991)-rw*t4*(t1055+t1132)))
		et2 = Idxx*t72*(Idxx*t72*(-t1342+t1611+rw*t4*(t1116-t1380+t9*t116*(t851-t1154)))+Idxx*t6*t73*(rf*t293*t1228-rw*t4*t1598+t6*t9*t115*(t701-t1031)))+rf*t6*t8*(t11*t349*(Idxx*(t1228*(t122-t208)+rw*t4*t1693+t6*t9*t115*t1593)+Idxx*t6*t73*(t413*t639+rw*t4*t946))+t6*t8*t178*((t122-t208)*(Idxx*(t912+Iwyy*(t840-t1019))-t72*t76*t803)-rw*t4*(t1592+t2035)+t6*t9*t115*(Idxx*t1700+t6*t73*t76*t803)))
		et3 = -Idxx*t6*t73*(Iwyy*(Idxx*t72*(t776*(t122-t208)+rf*t293*(t677+t936))+t6*t73*t76*(t817-t1225))-rw*t4*(Idxx*t72*(-t859+t1255+t254*(t677+t936))+Idxx*t6*t73*t1591))
		et4 = Idxx*(Iwyy*t2392-t71*t2270+rw*t9*(-t1909+t2121+t254*(t338*(t995+t1127)+(t544*(t840-t1019))/2.0+t178*t252*(t1211+t1350))))+t72*t76*(Idxx*t72*(t1474+Iwyy*(t851-t1125-t1154+t1384)+rw*t9*(t1116-t1380+t9*t116*(t851-t1154)))-Idxx*t6*t73*(-Iwyy*t1738+t71*t979+rw*t9*t1598))
		et5 = t11*t115*(Idxx*(t11*t349*(t1474+Iwyy*(t851-t1125-t1154+t1384)+rw*t9*(t1116-t1380+t9*t116*(t851-t1154)))+t6*t8*t178*(Iwyy*t1956-t71*t1631+rw*t9*(-t1072+t1549+t254*(t995+t1127))))-t6*t73*t76*(Idxx*t6*t73*(Iwyy*t1168-rw*t9*(t254*t723-t9*t116*t724))+t6*t8*t72*t76*t178*t1423))
		et6 = -rf*t6*t8*(t11*t349*(Idxx*(-Iwyy*t1891+t71*t1131+rw*t9*t1693)+Idxx*t6*t73*(Iwyy*t1064+rw*t9*t946))+t6*t8*t178*(Idxx*(-t71*t1515+Iwyy*(t1211+t1350+t1397+t4*t6*t115*(t995+t1127))+rw*t9*(t334*(t840-t1019)+t9*t116*(t1211+t1350)))+Idxx*t72*(Idxx*t72*(t373-t689+t1016+t1299)-t6*t73*t76*t1337)-t6*t73*t76*(Iwyy*t1550-rw*t9*(Idxx*t72*(t525+t893)+t9*t116*t902))))-Idxx*t6*t73*(Iwyy*t2047+rw*t9*(Idxx*t72*(-t859+t1255+t254*(t677+t936))+Idxx*t6*t73*t1591))
		et7 = t76*(-t71*(-t1746+t1844+t1660*(t122-t208))+rw*t9*(t1997+t1888*(t122-t208)+rf*t293*(t338*(t995+t1127)+(t544*(t840-t1019))/2.0+t178*t252*(t1211+t1350)))+rw*t4*t2392)+t11*t115*(Idxx*(t11*t349*(t1875+t71*(t854+t603*(t122-t208))+rw*t9*(t1387+t1022*(t122-t208)))-t6*t8*t178*(-t71*(t1101-t609*(t122-t208))+rw*t9*(t970*(t122-t208)+rf*t293*(t995+t1127))+rw*t4*t1956))-Idxx*t6*t73*(Idxx*t6*t73*(rw*t4*t1168-rf*rw*t9*t293*t723)+t6*t8*t72*t76*t178*(rw*t4*(t360+t524/2.0+rf*t252*t507)-rf*t293*t559)))
		et8 = -Idxx*t72*(Idxx*t72*(t1875+t71*(t854+t603*(t122-t208))+rw*t9*(t1387+t1022*(t122-t208)))+t6*t73*t76*(t71*(t655-t671)-rw*t9*(rf*t293*t1085-t6*t9*t115*t1022)+rw*t4*t1738))+Idxx*t6*t73*(rw*t9*(Idxx*t72*(t776*(t122-t208)+rf*t293*(t677+t936))+t6*t73*t76*(t817-t1225))+rw*t4*t2047)
		et9 = -rf*t6*t8*(t11*t349*(Idxx*(t71*(t818+t583*(t122-t208))+rw*t9*(t1085*(t122-t208)+t6*t9*t115*t1348)+rw*t4*t1891)-Idxx*t6*t73*(rw*t4*t1064-rw*t9*t178*t252*t639))-t6*t8*t178*(t76*(t71*t1581-rw*t4*(t1211+t1350+t1397+t4*t6*t115*(t995+t1127))+rw*t9*((t122-t208)*(t840-t1019)-t6*t9*t115*(t995+t1127)))-Idxx*t72*(Idxx*t72*(t559*(t122-t208)+rw*t4*(t989+t336*t507))-Idxx*t6*t73*(rw*t4*t945+t6*t9*t115*t559))+Idxx*t6*t73*(rw*t4*t1550-rw*t4*t6*t9*t349*t639)))
		et10 = t768*t2502*(et1+et2+et3)-t771*t2502*(et4+et5+et6)+t1754*t2506*(Idxx*(t1815-t2242+Iwyy*(t1912+t1738*(t122-t208)+t6*t9*t115*(t851-t1125-t1154+t1384))+rw*t4*(t1759-t1905+t9*t116*(t1249-t1455+t4*t6*t115*(t851-t1154)))+t6*t8*t178*(t11*t115*(t1690+t1423*(t122-t208)+rw*t4*(t717-t969-t1070+t1320))+rf*t6*t8*(t1579+t1337*(t122-t208)+t6*t9*t115*(t373-t689+t1016+t1299))))-t6*t73*t76*t2197)
		et11 = -t1540*t2504*(Idxx*t2455+t11*t349*(t11*t115*(Idxx*(t1690+t1423*(t122-t208)+rw*t4*(t717-t969-t1070+t1320))-t27*t55*t110*t687)+rf*t6*t8*(Idxx*(t1579+t1337*(t122-t208)+t6*t9*t115*(t373-t689+t1016+t1299))-Idxx*t6*t73*(Idxx*t72*t907+t6*t73*t76*t562)))-t72*t76*t2337-t6*t73*t76*t2260)+t788*t2502*(et7+et8+et9)+t1810*t2504*(Idxx*t2449+t72*t76*t2197)
		et12 = -t471*t2502*(t11*t349*(Idxx*(t1815-t2242+Iwyy*(t1912+t1738*(t122-t208)+t6*t9*t115*(t851-t1125-t1154+t1384))+rw*t4*(t1759-t1905+t9*t116*(t1249-t1455+t4*t6*t115*(t851-t1154))))+t6*t73*t76*(Idxx*t72*(t1475+Iwyy*(t854+t603*(t122-t208))+rw*t9*(t886+t681*(t122-t208)))-Idxx*t6*t73*(t934+t1005+Iwyy*(t655-t671))))-t6*t8*t178*(t2261+t2338+t2456))
		et13 = -t2503*(Idxx*t72*(t1815-t2242+Iwyy*(t1912+t1738*(t122-t208)+t6*t9*t115*(t851-t1125-t1154+t1384))+rw*t4*(t1759-t1905+t9*t116*(t1249-t1455+t4*t6*t115*(t851-t1154)))+t6*t8*t178*(t11*t115*(t1690+t1423*(t122-t208)+rw*t4*(t717-t969-t1070+t1320))+rf*t6*t8*(t1579+t1337*(t122-t208)+t6*t9*t115*(t373-t689+t1016+t1299))))-t6*t73*t76*t2449)*(t34-t283+t369+t519+t553+t707)
		et14 = Idxx*(t71*t2272+Iwyy*(t1482+t1687+t1923+t2003+t2106+(t60*t73*t112*t1529)/2.0)+rw*t4*t2374)+t11*t115*(Idxx*(t11*t349*(t1499-t1893+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154)))+t6*t8*t178*t2300)-Idxx*t6*t73*(Idxx*t6*t73*(Iwyy*(t459-(t349*t436)/2.0+rf*t293*(t498-t552))-rw*t4*(t254*(t498-t552)+t4*t116*t724))-t6*t8*t72*t76*t178*t1448))+t72*t76*(Idxx*t72*(t1499-t1893+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154)))-Idxx*t6*t73*t2062)
		et15 = -rf*t6*t8*(t11*t349*(Idxx*t2180-t6*t73*t76*(Iwyy*t1088+rw*t4*t951))+t6*t8*t178*(-Idxx*t2314+t72*t76*(Idxx*t72*(t400+t1004+t619*(t122-t208)+rw*t4*(t1038+t334*(t285-t414)))-Idxx*t6*t73*(t860+t960))+Idxx*t6*t73*(Iwyy*(Idxx*t72*(t408+t746+t6*t9*t349*(t122-t208))-Idxx*t6*t73*t1162)+rw*t4*(Idxx*t72*t1041-(Idxx*t4*t6*t73*t116*t869)/4.0))))+t6*t73*t76*(Iwyy*t2052+rw*t4*t2007)
		et16 = t76*(-t71*t2254+rw*t9*(t1482+t1687+t1923+t2003+t2106+(t60*t73*t112*t1529)/2.0)+rw*t4*t2359)+t11*t115*(Idxx*(t11*t349*t2159+t6*t8*t178*(t336*(t625+t1018)+rf*t252*(t1057+t1502)-rw*t9*t1961))-Idxx*t6*t73*(rw*t9*(-t834+rf*t293*(t578+t6*t8*t178*(Idxx*t72*(t285-t414)+t9*t55*t73*t76*t349))+Idxx*t6*t73*t563)-rf*t252*(rw*t4*(t578+t6*t8*t178*(Idxx*t72*(t285-t414)+t9*t55*t73*t76*t349))+t6*t8*t71*t72*t76*t178)))-Idxx*t72*(Idxx*t72*t2159-t6*t73*t76*(rf*t252*t1319-rw*t9*t1749+t4*t6*t115*t1316))+Idxx*t6*t73*(rw*t4*t1809+rw*t9*t2052)
		et17 = rf*t6*t8*(t11*t349*(Idxx*(t336*t1319-rw*t9*t1904+t4*t6*t115*t1594)+Idxx*t6*t73*(rw*t9*t1088-rw*t4*t178*t293*t653))-t6*t8*t178*(t336*(Idxx*(t1176+t1363)+t72*t76*t779)+rw*t9*(-t1599+t2031+t6*t9*t115*(t758-Idxx*t1486))-t4*t6*t115*(Idxx*(t1057+t1502)-Idxx*t6*t73*t779)))
		et18 = -Idxx*(Iwyy*t2359+t71*t2189-rw*t9*t2374)+t11*t115*(Idxx*(t11*t349*(t1617+t336*(t708+Iwyy*(t751+(t409*(t285-t414))/2.0))+rw*t9*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154)))-t6*t8*t178*(t336*(t988-t71*(t493-(t4*t60*t73*t112*t116)/2.0))-rw*t9*t1964+rf*t252*(t1071-t1494)))-Idxx*t6*t73*(rw*t9*(t1117+t254*(t578+t6*t8*t178*(Idxx*t72*(t285-t414)+t9*t55*t73*t76*t349)))-rf*t252*(t451+t1014)))
		et19 = -Idxx*t72*(Idxx*t72*(t1617+t336*(t708+Iwyy*(t751+(t409*(t285-t414))/2.0))+rw*t9*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154)))-Idxx*t6*t73*(-rf*t252*t1240+rw*t9*t1602+t4*t6*t115*(t708+Iwyy*(t751+(t409*(t285-t414))/2.0))))
		et20 = rf*t6*t8*(t11*t349*(Idxx*(t336*t1240-rw*t9*t1703+t4*t6*t115*t1600)-Idxx*t6*t73*(rw*t9*t951+Iwyy*t178*t293*t653))+t6*t8*t178*(t336*(Idxx*(t911-t1358)+t72*t76*(t307+Iwyy*(Idxx*t72*(t285-t414)+t9*t55*t73*t76*t349)))+rw*t9*(t2034-t334*(t738-Idxx*t1351))-t4*t6*t115*(Idxx*(t1071-t1494)+t6*t73*t76*(t307+Iwyy*(Idxx*t72*(t285-t414)+t9*t55*t73*t76*t349)))))+Idxx*t6*t73*(Iwyy*t1809-rw*t9*t2007)
		et21 = t768*t2502*(et14+et15)-t1754*t2506*(Idxx*(-t2078+t2194+t2345+rw*t9*(t1778+t1919+t1950)+t4*t6*t115*(t1499-t1893+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154))))+t6*t73*t76*t2171)+t788*t2502*(et16+et17)-t1810*t2504*(Idxx*t2448+Idxx*t72*t2171)+t471*t2502*(t11*t349*(Idxx*(-t2078+t2194+rw*t9*(t1778+t1919+t1950)+t4*t6*t115*(t1499-t1893+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154))))+Idxx*t6*t73*(Idxx*t72*t1834+Idxx*t6*t73*(t810+t1012+t1050)))-t6*t8*t178*(t2253-Idxx*t2453+t72*t76*t2339))
		et22 = t2503*(Idxx*t72*(-t2078+t2194+t2345+rw*t9*(t1778+t1919+t1950)+t4*t6*t115*(t1499-t1893+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154))))+Idxx*t6*t73*t2448)*(t34-t283+t369+t519+t553+t707)+t771*t2502*(et18+et19+et20)+t1540*t2504*(t2253-Idxx*t2453+t72*t76*t2339+t11*t349*(t11*t115*(Idxx*t2183+t27*t55*t110*(t454-t535))-rf*t6*t8*(Idxx*t2089-t6*t73*t76*(Idxx*t72*t901+t6*t73*t76*t560))))
		et23 = t471*t2502*(t2435+t76*t2494+Idxx*t72*(t2395+Idxx*t72*(t1499-t1893-t2028+t2144-t2276+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154)))))-t1754*t2506*(t11*t115*(t1643+Idxx*(t1499-t1893-t2028+t2144-t2276+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154))))+rf*t6*t8*t2415)+t2503*(t11*t115*(t2391+Idxx*t72*(t1499-t1893-t2028+t2144-t2276+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154))))+rf*t6*t8*(t2406+t6*t73*t76*t2398))*(t34-t283+t369+t519+t553+t707)
		et24 = t1810*t2504*(t11*t115*(t1644-t2389)+rf*t6*t8*(t1951+t2399))+t788*t2502*(t11*t115*(t2112+t2412)-rf*t6*t8*(t2136+t2248+t2427))-t768*t2502*(t11*t115*(t2200+t2431)-rf*t6*t8*(t2219+t2297+t2444))-t771*t2502*(t11*t115*(t2201+t2430)-rf*t6*t8*(t2221+t2299+t2446))+t1540*t2504*(t11*t115*(Idxx*(t1639-t1962-t1969+t2210+t2309+t336*(t641-t978-t1654+rf*t293*(t647-t935)))-t6*t73*t76*t2088)+rf*t6*t8*t2463)
		et25 = t788*t2502*(Idxx*(-t2040+t2258+t2285-t2289+t11*t115*(t1826-t1865))+Idxx*t72*(t2100+rf*t6*t8*t1496))-t1540*t2504*(Idxx*(-t2158+t2231+t2239+rf*t252*(t912+Iwyy*(t840-t1019)+rw*t4*(t1372+t4*t116*(t840-t1019))+t6*t9*t115*(t647-t935))+t4*t6*t115*(t641-t978-t1654+rf*t293*(t647-t935)))-t72*t76*(t2088+Idxx*rf*t8*t11*t55*t73*t349*t432))-t1754*t2506*(Idxx*t2410-t6*t27*t72*t73*t1653)+t471*t2502*(Idxx*t2462+t72*t76*t2229)+t1810*t2504*(Idxx*t2401-t27*t131*t1653)
		et26 = -t771*t2502*(Idxx*t2461-t72*t76*(Idxx*t72*(t708+t1288+Iwyy*(t751+(t409*(t285-t414))/2.0)+rw*t9*(-t974+t1067+t9*t116*(t751+(t409*(t285-t414))/2.0)))+rf*t6*t8*t1713+t6*t73*t76*t1734))+t2503*(t2403+Idxx*t72*t2410)*(t34-t283+t369+t519+t553+t707)-t768*t2502*(Idxx*t2459+Idxx*t72*(t2196+rf*t6*t8*t1712))
		et27 = t771*t2502*(t2201+t2430+rf*t6*t8*(Idxx*(t11*t349*(t830+t1305+Iwyy*(t1134+(t544*(t285-t414))/2.0)+rw*t9*(-t994+t1370+t9*t116*(t1134+(t544*(t285-t414))/2.0)))+t6*t8*t178*(t1071-t1494+t1938+t336*(t647-t935)))+t6*t73*t76*t1713))-t1810*t2504*(t1644-t2389+rf*t6*t8*(t577-t2127))+t1754*t2506*(t1643+Idxx*(t1499-t1893-t2028+t2144-t2276+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154)))+rf*t55*t63*t178*t2055)
		et28 = -t2503*(t2391+Idxx*t72*(t1499-t1893-t2028+t2144-t2276+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154)))+rf*t6*t8*t2323)*(t34-t283+t369+t519+t553+t707)-t1540*t2504*(t2091-t76*(t1639-t1962-t1969+t2210+t2309+t336*(t641-t978-t1654+rf*t293*(t647-t935)))+rf*t6*t8*t11*t349*t2055)-t788*t2502*(t2112+t2412+rf*t6*t8*t2341)+t768*t2502*(t2200+t2431+rf*t6*t8*t2387)
		et29 = -t471*t2502*(t2233+Idxx*(t2424+t11*t349*(t1499-t1893-t2028+t2144-t2276+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154)))))
		et30 = t1540*t2504*(t2146+Idxx*(-t2207+t2303+t2314+t336*(t912+Iwyy*(t840-t1019)+rw*t4*(t1372+t4*t116*(t840-t1019))+t6*t9*t115*(t647-t935)))+t72*t76*t2237+t69*t115*t349*t2055)-t771*t2502*(t2221+t2299+t2446+t11*t115*(Idxx*(t11*t349*(t830+t1305+Iwyy*(t1134+(t544*(t285-t414))/2.0)+rw*t9*(-t994+t1370+t9*t116*(t1134+(t544*(t285-t414))/2.0)))+t6*t8*t178*(t1071-t1494+t1938+t336*(t647-t935)))+t6*t73*t76*t1713))-t1754*t2506*(t1949+t76*t2404+t6*t8*t11*t115*t178*t2055)-t1810*t2504*(t1951+t2399-t11*t115*(t577-t2127))
		et31 = t2503*(t2400+t11*t115*t2323+t72*t76*t2404)*(t34-t283+t369+t519+t553+t707)-t471*t2489*t2502+t788*t2502*(t2136+t2248+t2427+t11*t115*t2341)-t768*t2502*(t2219+t2297+t2444+t11*t115*t2387)
		et32 = Idxx*(-t1909+t2121-t2384+t254*(t338*(t995+t1127)+(t544*(t840-t1019))/2.0+t178*t252*(t1211+t1350))+t2324*(t122-t208)+rf*t293*(-t1873+t2096+t4*t116*(t338*(t995+t1127)+(t544*(t840-t1019))/2.0+t178*t252*(t1211+t1350))))-t72*t76*(Idxx*t72*t2275+Idxx*t6*t73*t2125)-t11*t115*(Idxx*(t11*t349*t2275-t6*t8*t178*(-t1072+t1549+t254*(t995+t1127)+t1651*(t122-t208)+rf*t293*(-t981+t1509+t4*t116*(t995+t1127))))+t6*t73*t76*(t1055+t1132+rf*t293*t1463))
		et33 = t6*t73*t76*(Idxx*t72*(-t859+t1255+t254*(t677+t936)+t1461*(t122-t208)+rf*t293*(-t877+t1274+t4*t116*(t677+t936)))-t6*t73*t76*t2068)+rf*t6*t8*(t11*t349*(Idxx*(-t1177+t1443+t1560*(t122-t208)+t6*t9*t115*(-t994+t1370+t9*t116*(t1134+(t544*(t285-t414))/2.0)))+t6*t73*t76*(Idxx*t72*(t801+t637*(t122-t208))-Idxx*t6*t73*t820))+t6*t8*t178*(t1592+t2035-t1862*(t122-t208)+t6*t9*t115*t2045))
		et34 = Idxx*t2471-t72*t76*(Idxx*t72*t2202-Idxx*t6*t73*(t1749+rf*t252*t1447+t4*t6*t115*t1437))-t11*t115*(Idxx*(t11*t349*t2202-t6*t8*t178*(t1961+t336*t1411-rf*t252*t1670))-t6*t73*t76*(t1081+t1091+rf*t252*t984))+rf*t6*t8*(t11*t349*(Idxx*(t1904+t336*t1447-t4*t6*t115*(-t761+t1121+t932*(t122-t208)))+t6*t73*t76*(Idxx*t72*(t545+t770)+Idxx*t6*t73*(t486+t583)))+t6*t8*t178*(-t1599+t2031-t336*(-t741+Idxx*t956+t76*(t840-t1019))+t6*t9*t115*(t758-Idxx*t1486)+t4*t6*t115*(-Idxx*(t995+t1127)+Idxx*t920*(t122-t208)+t6*t73*t76*t704)))
		et35 = t6*t73*t76*(Idxx*t72*t2076+t6*t73*t76*t2038)
		et36 = Idxx*t2476+t11*t115*(Idxx*(t11*t349*(t1145+t1701+t254*(t1134+(t544*(t285-t414))/2.0)+rf*t252*(-t994+t1370+t9*t116*(t1134+(t544*(t285-t414))/2.0))+t4*t116*(t851-t1154))+t6*t8*t178*(t1964+t336*t1651-rf*t252*(-t981+t1509+t4*t116*(t995+t1127))))-Idxx*t6*t73*(t1117+t254*(t578+t6*t8*t178*(Idxx*t72*(t285-t414)+t9*t55*t73*t76*t349))+rf*t252*t1463))
		et37 = -Idxx*t72*(Idxx*t72*(t1145+t1701+t254*(t1134+(t544*(t285-t414))/2.0)+rf*t252*(-t994+t1370+t9*t116*(t1134+(t544*(t285-t414))/2.0))+t4*t116*(t851-t1154))-t6*t73*t76*(-t1172+t1219+rf*t252*t1560+t4*t6*t115*(-t974+t1067+t9*t116*(t751+(t409*(t285-t414))/2.0))))
		et38 = rf*t6*t8*(t11*t349*(Idxx*(-t1196+t1442+t336*t1560+t4*t6*t115*(-t994+t1370+t9*t116*(t1134+(t544*(t285-t414))/2.0)))+t6*t73*t76*(Idxx*t72*(t811+t336*t637)+t6*t73*t76*(t4*t6*t115*t637-(t4*t60*t73*t112*t116)/2.0)))+t6*t8*t178*(t2034-t336*t1862-t334*(t738-Idxx*t1351)+t4*t6*t115*t2045))+t6*t73*t76*(Idxx*t72*t2164+t6*t73*t76*(t1487-t1595+t1621))
		et39 = -t768*t2502*(et32+et33)+t471*t2502*(t11*t349*(Idxx*(t1778+t1919+t1950+t2132+t2225+t2280)+t6*t73*t76*(Idxx*t72*(-t548+t618+t895+t1126+t693*(t122-t208)+rf*t252*(t801+t637*(t122-t208)))+t6*t73*t76*(t654+t727+t894+t1087)))-t6*t8*t178*(t2227-t76*(-t1857+t2064+t2235+rf*t252*(-t1781+t334*(t840-t1019)+t9*t116*(t1211+t1350)+t6*t9*t115*(-t981+t1509+t4*t116*(t995+t1127)))+t4*t116*(-t1554+t1375*(t122-t208)+rf*t293*(t1211+t1350))+t4*t6*t115*(-t1072+t1549+t254*(t995+t1127)+t1651*(t122-t208)+rf*t293*(-t981+t1509+t4*t116*(t995+t1127))))+Idxx*t72*(t1931-Idxx*t72*t2131)))
		et40 = -t788*t2502*(et34+et35)-t1540*t2504*(t2228+t2292+t2452-t11*t349*(t11*t115*(Idxx*t2131-t27*t55*t110*t695)-rf*t6*t8*(Idxx*t2009-t6*t27*t72*t73*t683)))+t2503*(Idxx*t72*t2458+t6*t73*t76*(-t1691-t1833+t1845+t2080-t2152+rf*t252*(-t1232+t1418+t1606*(t122-t208)+t6*t9*t115*(-t877+t1274+t4*t116*(t677+t936)))+t11*t115*(t1965+t11*t349*(-t548+t618+t895+t1126+t693*(t122-t208)+rf*t252*(t801+t637*(t122-t208))))+t4*t6*t115*(-t859+t1255+t254*(t677+t936)+t1461*(t122-t208)+rf*t293*(-t877+t1274+t4*t116*(t677+t936)))))*(t34-t283+t369+t519+t553+t707)
		et41 = t771*t2502*(et36+et37+et38)+t1810*t2504*(Idxx*(-t1691-t1833+t1845+t2080-t2152+rf*t252*(-t1232+t1418+t1606*(t122-t208)+t6*t9*t115*(-t877+t1274+t4*t116*(t677+t936)))+t11*t115*(t1965+t11*t349*(-t548+t618+t895+t1126+t693*(t122-t208)+rf*t252*(t801+t637*(t122-t208))))+t4*t6*t115*(-t859+t1255+t254*(t677+t936)+t1461*(t122-t208)+rf*t293*(-t877+t1274+t4*t116*(t677+t936))))-t72*t76*t2151)-t1754*t2506*(Idxx*t2458-t6*t73*t76*t2151)
		et42 = t788*t2502*(Idxx*t72*(t2411+rf*t6*t8*t2311)+Idxx*t6*t73*(-t2040+t2258+t2285-t2289+t11*t115*(t1826-t1865)))-t1540*t2504*(Idxx*t72*(t2422+rf*t6*t8*t11*t349*(t400+t1004+t1151+t619*(t122-t208)+rw*t9*(-t597+t1039+t800*(t122-t208))+rw*t4*(t1038+t334*(t285-t414))))-t6*t73*t76*(-t2158+t2231+t2239+rf*t252*(t912+Iwyy*(t840-t1019)+rw*t4*(t1372+t4*t116*(t840-t1019))+t6*t9*t115*(t647-t935))+t4*t6*t115*(t641-t978-t1654+rf*t293*(t647-t935))))
		et43 = -t771*t2502*(Idxx*t72*(t2429+rf*t6*t8*(t11*t349*(t830+t1305+Iwyy*(t1134+(t544*(t285-t414))/2.0)+rw*t9*(-t994+t1370+t9*t116*(t1134+(t544*(t285-t414))/2.0)))+t6*t8*t178*(t1071-t1494+t1938+t336*(t647-t935))))-t6*t73*t76*t2461)-t1810*t2504*(t2403+Idxx*t72*(t2388+rf*t6*t8*t2126))
		et44 = t2503*(t2494+t11*t115*(t2424+t11*t349*(t1499-t1893-t2028+t2144-t2276+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154))))+rf*t6*t8*(t11*t349*t2404+t6*t8*t178*(-t2207+t2303+t2314+t336*(t912+Iwyy*(t840-t1019)+rw*t4*(t1372+t4*t116*(t840-t1019))+t6*t9*t115*(t647-t935)))))*(t34-t283+t369+t519+t553+t707)
		et45 = t768*t2502*(Idxx*t72*(t2379+Iwyy*(-t1215+t1575+(t409*(t995+t1127))/2.0+t1709*(t122-t208)+rf*t293*(-t1160+t1551+t178*t293*(t995+t1127)))+t71*(-t859+t1255+t254*(t677+t936)+t1461*(t122-t208)+rf*t293*(-t877+t1274+t4*t116*(t677+t936)))-rf*t6*t8*t2362)-Idxx*t6*t73*t2459)+t471*t2502*(Idxx*t72*(t2424+t11*t349*(t1499-t1893-t2028+t2144-t2276+rw*t4*(t1145+t254*(t1134+(t544*(t285-t414))/2.0)+t4*t116*(t851-t1154))))+Idxx*t6*t73*t2462)
		et46 = t1754*t2506*(Idxx*t72*(t2417+rf*t55*t63*t178*(t400+t1004+t1151+t619*(t122-t208)+rw*t9*(-t597+t1039+t800*(t122-t208))+rw*t4*(t1038+t334*(t285-t414))))-Idxx*t6*t73*t2410)

		dx = th.zeros((x.shape[0], 16), device=self.device, dtype=self.th_dtype)
		dx[:,0] = vx
		dx[:,1] = vy
		dx[:,2] = vz
		dx[:,3] = v_ph
		dx[:,4] = v_th
		dx[:,5] = v_om
		dx[:,6] = v_ps - v_om
		dx[:,7] = v_ne

		dx[:,8] = et10+et11+et12+et13
		dx[:,9] = et21+et22
		dx[:,10] = et23+et24
		dx[:,11] = et25+et26
		dx[:,12] = et27+et28+et29
		dx[:,13] = et30+et31
		dx[:,14] = et39+et40+et41 - (et30+et31)
		dx[:,15] = et42+et43+et44+et45+et46

		return dx
	
	def _enforce_bounds(self, state_):
		# bound the independent dims and compute the dependent dims accordingly!
		state_[:, self.independent_dims] = th.minimum(self.th_x_bounds[self.independent_dims,1], th.maximum(self.th_x_bounds[self.independent_dims,0], state_[:,self.independent_dims]))
		err_p, err_v = self._err_posvel_contact(state_)
		state_[:, :3] += (-err_p)
		state_[:, 8:11] += (-err_v)

		return state_
	
	def _err_posvel_contact(self, x):

		Rframe = th_get_rotation(x[:,5], x[:,4], x[:,3], device=self.device, dtype=self.th_dtype)
		Rwheel = th_get_rotation(th.zeros(x.shape[0], device=self.device, dtype=self.th_dtype), x[:,4], x[:,3], device=self.device, dtype=self.th_dtype)
		err_p = x[:,:3] + Rframe @ th.asarray([0., 0., -self.rf], device=self.device, dtype=self.th_dtype) + Rwheel @ th.asarray([0., 0., -self.rw], device=self.device, dtype=self.th_dtype)
		err_p[:,:2] = 0.
		c_J = self._jacobian_contact(x)
		err_v = th.squeeze(th.bmm(c_J, th.unsqueeze(x[:, 8:], dim=2)), dim=2)

		return err_p, err_v
	
	def _jacobian_com(self, x):

		mw = self.mw
		mf = self.mf
		md = self.md

		rw = self.rw
		rf = self.rf
		rd = self.rd

		ph = x[:,3]
		theta = x[:,4]
		om = x[:,5]
		ps = x[:,6] + om
		ne = x[:,7]

		J = th.zeros((self.num_envs, 3, 8), device=self.device, dtype=self.th_dtype)
		J[:,0,0] = 1.
		J[:,0,3] = ((md*rd - mw*rf)*(th.cos(ph)*th.cos(om)*th.sin(theta) - th.sin(ph)*th.sin(om)))/(md + mf + mw)
		J[:,0,4] = ((md*rd - mw*rf)*th.cos(theta)*th.cos(om)*th.sin(ph))/(md + mf + mw)
		J[:,0,5] = ((md*rd - mw*rf)*(th.cos(ph)*th.cos(om) - th.sin(theta)*th.sin(ph)*th.sin(om)))/(md + mf + mw)

		J[:,1,1] = 1.
		J[:,1,3] = ((md*rd - mw*rf)*(th.cos(om)*th.sin(theta)*th.sin(ph) + th.cos(ph)*th.sin(om)))/(md + mf + mw)
		J[:,1,4] = ((-(md*rd) + mw*rf)*th.cos(theta)*th.cos(ph)*th.cos(om))/(md + mf + mw)
		J[:,1,5] = ((md*rd - mw*rf)*(th.cos(om)*th.sin(ph) + th.cos(ph)*th.sin(theta)*th.sin(om)))/(md + mf + mw)

		J[:,2,2] = 1.
		J[:,2,4] = ((-(md*rd) + mw*rf)*th.cos(om)*th.sin(theta))/(md + mf + mw)
		J[:,2,5] = ((-(md*rd) + mw*rf)*th.cos(theta)*th.sin(om))/(md + mf + mw)

		J[:,:,5] += J[:,:,6]

		return J
	
	def _jacobian_contact_trace(self, x):
		
		mw = self.mw
		mf = self.mf
		md = self.md

		rw = self.rw
		rf = self.rf
		rd = self.rd

		ph = x[:,3]
		theta = x[:,4]
		om = x[:,5]
		ps = x[:,6] + om
		ne = x[:,7]

		J = th.zeros((self.num_envs, 3, 8), device=self.device, dtype=self.th_dtype)
		J[:,0,0] = 1.
		J[:,0,3] = -(th.cos(ph)*(rw + rf*th.cos(om))*th.sin(theta)) + rf*th.sin(ph)*th.sin(om)
		J[:,0,4] = -(th.cos(theta)*(rw + rf*th.cos(om))*th.sin(ph))
		J[:,0,5] = -(rf*th.cos(ph)*th.cos(om)) + rf*th.sin(theta)*th.sin(ph)*th.sin(om)
		
		J[:,1,1] = 1.
		J[:,1,3] = -((rw + rf*th.cos(om))*th.sin(theta)*th.sin(ph)) - rf*th.cos(ph)*th.sin(om)
		J[:,1,4] = th.cos(theta)*th.cos(ph)*(rw + rf*th.cos(om))
		J[:,1,5] = -(rf*(th.cos(om)*th.sin(ph) + th.cos(ph)*th.sin(theta)*th.sin(om)))

		J[:,2,2] = 1.
		J[:,2,4] = (rw + rf*th.cos(om))*th.sin(theta)
		J[:,2,5] = rf*th.cos(theta)*th.sin(om)

		J[:,:,5] += J[:,:,6]

		return J

	def _jacobian_contact(self, x):
		rw = self.rw
		rf = self.rf
		ph = x[:, 3]
		theta = x[:, 4]
		om = x[:, 5]
		ps = x[:, 6] + om
		ne = x[:, 7]

		J = th.zeros((self.num_envs, 3, 8), device=self.device, dtype=self.th_dtype)
		J[:,0,0] = 1.
		J[:,0,3] = -(th.cos(ph)*(rw + rf*th.cos(om))*th.sin(theta)) + rf*th.sin(ph)*th.sin(om)
		J[:,0,4] = -(th.cos(theta)*(rw + rf*th.cos(om))*th.sin(ph))
		J[:,0,5] = -(rf*th.cos(ph)*th.cos(om)) + rf*th.sin(theta)*th.sin(ph)*th.sin(om)
		J[:,0,6] = -(rw*th.cos(ph))

		J[:,1,1] = 1.
		J[:,1,3] = -((rw + rf*th.cos(om))*th.sin(theta)*th.sin(ph)) - rf*th.cos(ph)*th.sin(om)
		J[:,1,4] = th.cos(theta)*th.cos(ph)*(rw + rf*th.cos(om))
		J[:,1,5] = -(rf*(th.cos(om)*th.sin(ph) + th.cos(ph)*th.sin(theta)*th.sin(om)))
		J[:,1,6] = -(rw*th.sin(ph))

		J[:,2,2] = 1.
		J[:,2,4] = (rw + rf*th.cos(om))*th.sin(theta)
		J[:,2,5] = rf*th.cos(theta)*th.sin(om)

		# v = ... + J[:,5] * v_om + J[:,6] * (v_ps + v_om)
		# v = ... + (J[:,5] + J[:,6]) * v_om + J[:,6] * v_ps
		J[:,:,5] += J[:,:,6]

		return J
	
	def _update_visualizer(self):
		#TODO visualize all the agents in parallel
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

	def render(self, model='human'):
		if (not (hasattr(self, 'viz') and isinstance(self.viz, meshcat.Visualizer))):
			self._create_visualizer()
		self._update_visualizer()
		
	def close(self):
		pass