import gym
from gym import spaces
from ipdb import set_trace
import numpy as np

class Biped2D(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, sys, fixed_start=False, normalized_actions=False):
    super(Biped2D, self).__init__()
    # Define model paramters
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
    self.m = sys['m']
    self.I = sys['I']
    self.d = sys['d']
    self.df = sys['df']
    self.l0 = sys['l0']
    self.g = sys['g']
    self.fixed_start = fixed_start
    self.normalized_actions = normalized_actions
    self.reset()

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using continuous actions:
    if (normalized_actions):
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.U_DIMS,), dtype=np.float32)
    else:
        # self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.U_DIMS,), dtype=np.float32)
        self.action_space = spaces.Box(low=self.u_limits[:,0], high=self.u_limits[:,1], dtype=np.float32)    
    self.observation_space = spaces.Box(low=self.x_limits[:,0], high=self.x_limits[:,1], dtype=np.float32)

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
    # limit_violation = (observation > self.x_limits[:,1]).any() or (observation < self.x_limits[:,0]).any()
    limit_violation = False
    observation = np.minimum(self.x_limits[:,1], np.maximum(self.x_limits[:,0], observation))
    self.state = observation
    self.step_count += 1

    # if (limit_violation):
    #   reward -= 100

    if ((self.step_count >= self.horizon) or limit_violation):
      done = True
      info = {'terminal_state': self.state, 'step_count' : self.step_count}
    else:
      done = False
      info = {'terminal_state': np.array([]), 'step_count' : self.step_count}

    return observation, reward, done, info

  def reset(self, state=None):
    if (state is None):
        if (self.fixed_start):
            observation = np.array([0.9, 0.2+np.pi/2, 0.3, -0.4, np.pi/8, -0.2])
        else:
            observation = 0.5 * (self.x_limits[:,0] + self.x_limits[:,1]) \
                          + 0.2 * (np.random.rand(self.X_DIMS) - 0.5) * (self.x_limits[:,1] - self.x_limits[:,0])
    else:
        observation = state

    self.state = observation
    self.step_count = 0
    return observation  # reward, done, info can't be included
  
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
    m = self.m
    I = self.I
    d = self.d
    df = self.df
    l0 = self.l0
    g = self.g

    x1 = x[0,0]
    x2 = x[1,0]
    x3 = x[2,0]
    x4 = x[3,0]
    x5 = x[4,0]
    x6 = x[5,0]

    u1 = u[0,0]
    u2 = u[1,0]
    u3 = u[2,0]
    u4 = u[3,0]

    t2 = np.cos(x2)
    t3 = np.sin(x2)
    l2 = np.sqrt((x1*t2 + df)**2 + (x1*t3)**2)
    contact1 = x1 <= l0
    contact2 = l2 <= l0
    u1 = u1 * contact1
    u2 = u2 * contact2
    u3 = u3 * contact1
    u4 = u4 * contact2

    t4 = df**2
    t5 = x1**2
    t8 = 1.0/m
    t9 = -x5
    t10 = 1.0/x1
    t6 = t2**2
    t7 = t2*x1
    t13 = t9+x2
    t11 = df*t7*2.0
    t12 = df+t7
    t14 = np.cos(t13)
    t15 = np.sin(t13)
    t16 = t6-1.0
    t17 = t4+t5+t11
    t18 = 1.0/t17
    t19 = 1.0/np.sqrt(t17)
    t20 = t12*t19
    t22 = t5*t16*t18
    t21 = np.arccos(t20)
    t24 = -t22
    t23 = -t21
    t26 = np.sqrt(t24)
    t25 = t23+x5

    dx = np.zeros((self.X_DIMS, 1))
    dx[0,0] = t2*x3+t3*x4+d*t14*x6
    dx[1,0] = -t10*(-t2*x4+t3*x3+d*t15*x6)
    dx[2,0] = t8*(t2*u1+t20*u2+t3*t10*u3+t19*t26*u4)
    dx[3,0] = -g+t8*(t3*u1+t26*u2-t2*t10*u3-t12*t18*u4)
    dx[4,0] = x6
    dx[5,0] = (u3+u4+d*u2*np.cos(t25)+d*t14*u1+d*t10*t15*u3-d*t19*u4*np.sin(t25))/I
    
    return dx

  def render(self, mode='human'):
    pass

  def close (self):
    pass