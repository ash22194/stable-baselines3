import gym
from gym import spaces
from ipdb import set_trace
import numpy as np

class Quadcopter(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, sys, fixed_start=False, normalized_actions=False):
    super(Quadcopter, self).__init__()
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
    self.l = sys['l']
    self.g = sys['g']
    self.bk = sys['bk']
    self.I = sys['I']
    self.fixed_start = fixed_start
    self.normalized_actions = normalized_actions
    self.reset()

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when unp.sing continuous actions:
    if (normalized_actions):
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.U_DIMS,), dtype=np.float32)
    else:
        # self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.U_DIMS,), dtype=np.float32)
        self.action_space = spaces.Box(low=self.u_limits[:,0], high=self.u_limits[:,1], dtype=np.float32)

    self.observation_space = spaces.Box(low=self.x_limits[:,0], high=self.x_limits[:,1], dtype=np.float32)
    # self.observation_space = spaces.Box(low=-10, high=10, shape=(self.X_DIMS,), dtype=np.float32)

  def step(self, action):
    # If scaling actions use this
    if (self.normalized_actions):
        action = 0.5*((self.u_limits[:,0] + self.u_limits[:,1]) + action*(self.u_limits[:,1] - self.u_limits[:,0]))
    a = action[:,np.newaxis]
    x = self.state[:,np.newaxis]
    np.cost = sum((x - self.goal) * np.matmul(self.Q, x - self.goal)) + sum((a - self.u0) * np.matmul(self.R, a - self.u0))
    np.cost = np.cost * self.dt
    reward = -np.cost[0]

    observation = self.dyn_rk4(x, a, self.dt)
    observation = observation[:,0]
    # limit_violation = (observation > self.x_limits[:,1]).any() or (observation < self.x_limits[:,0]).any()
    # if (limit_violation):
    #     reward -= 1

    limit_violation = False
    # observation = np.minimum(self.x_limits[:,1], np.maximum(self.x_limits[:,0], observation))
    observation = np.minimum(10, np.maximum(-10, observation))
    self.state = observation
    self.step_count += 1

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
            observation = np.array([0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            observation = 0.5 * (self.x_limits[:,0] + self.x_limits[:,1]) \
                          + 0.4 * (np.random.rand(self.X_DIMS) - 0.5) * (self.x_limits[:,1] - self.x_limits[:,0])
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
  
  def dyn_full_rk4(self, x, u, dt):
    k1 = self.dyn_full(x, u)
    q = x + 0.5*k1*dt

    k2 = self.dyn_full(q, u)
    q = x + 0.5*k2*dt

    k3 = self.dyn_full(q, u)
    q = x + k3*dt

    k4 = self.dyn_full(q, u)
    
    q = x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0

    return q

  def dyn(self, x, u):
    m = self.m
    l = self.l
    g = self.g
    bk = self.bk
    II = self.I
    I1 = II[0,0]
    I2 = II[1,1]
    I3 = II[2,2]

    x2 = x[1,0]
    x3 = x[2,0]
    x4 = x[3,0]
    x7 = x[6,0]
    x8 = x[7,0]
    x9 = x[8,0]
    x10 = x[9,0]

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
    dx[0,0] = x7
    dx[1,0] = x8
    dx[2,0] = x9
    dx[3,0] = x10
    dx[4,0] = t23*u1*(t5*t7+t2*t4*t6)
    dx[5,0] = -t23*u1*(t4*t5-t2*t6*t7)
    dx[6,0] = -g+t2*t3*t23*u1
    dx[7,0] = (t21*t22*t24*(et1+et2))/I1
    dx[8,0] = mt2
    dx[9,0] = mt3
    
    return dx

  def dyn_full(self, x, u):
    
    dx = np.zeros((self.X_DIMS+2, 1))
    dx[0:2,:] = x[6:8,:] # vx, vy
    dx[2:,:] = self.dyn(x[2:,:], u)

    return dx

  def render(self, mode='human'):
    pass

  def close (self):
    pass