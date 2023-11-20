import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CartPole(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, sys=None, fixed_start=False, normalized_actions=True, expand_limits=False):
    # super(CartPole, self).__init__()
    # Define model paramters
    if (sys is None):
      sys = {'mc': 5, 'mp': 1, 'l': 0.9, 'g': 9.81, 'Q': np.diag([25, 0.02, 25, 0.02]), 'R': np.diag([0.001, 0.001]),\
	          'goal': np.array([[0], [0], [np.pi], [0]]), 'u0': np.zeros((2,1)), 'T': 4, 'dt': 2e-3, 'lambda_': 3, 'X_DIMS': 4, 'U_DIMS': 2,\
	          'x_limits': np.array([[-1.5, 1.5], [-3, 3], [0, 2*np.pi], [-3, 3]]), 'u_limits': np.array([[-9, 9], [-9, 9]])}
      sys['gamma_'] = np.exp(-sys['lambda_']*sys['dt'])

    self.X_DIMS = sys['X_DIMS']
    self.U_DIMS = sys['U_DIMS']
    self.goal = sys['goal']
    self.u0 = sys['u0']
    self.T = sys['T']
    self.dt = sys['dt']
    self.horizon = round(self.T / self.dt)
    self.x_limits = sys['x_limits'] # reset within these limits
    self.u_limits = sys['u_limits']
    self.Q = sys['Q']
    self.R = sys['R']
    self.mc = sys['mc']
    self.mp = sys['mp']
    self.l = sys['l']
    self.g = sys['g']
    self.fixed_start = fixed_start
    self.normalized_actions = normalized_actions
    self.expand_limits = expand_limits
    # self.reset()

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using continuous actions:
    # self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.U_DIMS,), dtype=np.float32)
    if (normalized_actions):
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.U_DIMS,), dtype=np.float32)
    else:
        self.action_space = spaces.Box(low=self.u_limits[:,0], high=self.u_limits[:,1], dtype=np.float32)
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.X_DIMS,), dtype=np.float32)

  def step(self, action):
    # If scaling actions use this
    if (self.normalized_actions):
        action = 0.5*((self.u_limits[:,0] + self.u_limits[:,1]) + action*(self.u_limits[:,1] - self.u_limits[:,0]))
    a = action[:,np.newaxis]
    x = self.state[:,np.newaxis]
    cost = sum((x - self.goal) * np.matmul(self.Q, x - self.goal)) + sum((a - self.u0) * np.matmul(self.R, a - self.u0))
    cost = cost * self.dt
    reward = -cost[0]

    observation = self.dyn_rk4(x, a, self.dt)
    observation = observation[:,0]
    reached_goal = np.linalg.norm(observation - self.goal[:,0]) < 1e-2
    # limit_violation = (observation >= self.x_limits[:,1]).any() or (observation <= self.x_limits[:,0]).any()
    limit_violation = False
    if (self.expand_limits):
      observation = np.minimum(10., np.maximum(-10., observation))
    else:
      observation = np.minimum(self.x_limits[:,1], np.maximum(self.x_limits[:,0], observation))
    self.state = observation
    self.step_count += 1

    # if (limit_violation):
    #   reward -= 100

    if ((self.step_count >= self.horizon) or limit_violation or reached_goal):
      done = True
      info = {'terminal_state': self.state, 'step_count' : self.step_count}
    else:
      done = False
      info = {'terminal_state': np.array([]), 'step_count' : self.step_count}

    return np.float32(observation), reward, done, False, info

  def reset(self, seed=None, options=None, state=None):
    super().reset(seed=seed)
    if (state is None):
        if (self.fixed_start):
            # observation = np.array([0.25, 0, 3*np.pi/4, 0.5])
            # observation = np.array([0, 0, np.pi, 0])
            observation = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            observation = 0.5 * (self.x_limits[:,0] + self.x_limits[:,1]) + 0.6 * (np.random.rand(self.X_DIMS) - 0.5) * (self.x_limits[:,1] - self.x_limits[:,0])
            observation = np.float32(observation)
    else:
        observation = state

    self.state = observation
    self.step_count = 0
    info = {'terminal_state': np.array([]), 'step_count' : self.step_count}

    return observation, info
  
  def dyn_rk4(self, x, u, dt):
    k1 = self.dyn(x, u)
    q = x + 0.5*k1*dt

    k2 = self.dyn(q, u)
    q = x + 0.5*k2*dt

    k3 = self.dyn(q, u)
    q = x + k3*dt

    k4 = self.dyn(q, u)
    
    q = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

    return q

  def dyn(self, x, u):
    mc = self.mc
    mp = self.mp
    l = self.l
    g = self.g

    x2 = x[1,0]
    x3 = x[2,0]
    x4 = x[3,0]

    u1 = u[0,0]
    u2 = u[1,0]

    t2 = np.cos(x3)
    t3 = np.sin(x3)
    t4 = x4**2
    t6 = 1.0/l
    t5 = t3**2
    t7 = mp*t5
    t8 = mc+t7
    t9 = 1.0/t8

    dx = np.zeros((self.X_DIMS, 1))
    dx[0,0] = x2
    dx[1,0] = t6*t9*(l*u1-t2*u2+mp*t3*t4*1.0/(t6**2)+g*l*mp*t2*t3)
    dx[2,0] = x4
    dx[3,0] = -t6*t9*(t2*u1-t6*u2*(mc/mp+1.0))-t6*t9*(g*t3*(mc+mp)+l*mp*t2*t3*t4)

    return dx

  def render(self, mode='human'):
    pass

  def close (self):
    pass