import gym
from gym import spaces
from ipdb import set_trace
import numpy as np

class LinearSystem(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, sys, add_quad_feat=False, normalized_actions=False):
    super(LinearSystem, self).__init__()
    # Define model paramters
    self.X_DIMS = sys['X_DIMS']
    self.U_DIMS = sys['U_DIMS']
    self.A = sys['A']
    self.B = sys['B']
    self.Q = sys['Q']
    self.R = sys['R']
    self.goal = sys['goal']
    self.u0 = sys['u0']
    self.T = sys['T']
    self.dt = sys['dt']
    self.horizon = round(self.T / self.dt)
    self.x_limits = sys['x_limits'] # reset within these limits
    self.u_limits = sys['u_limits']
    self.add_quad_feat = add_quad_feat
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

    if (add_quad_feat):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(int(1/2*self.X_DIMS*(self.X_DIMS + 3)),), dtype=np.float32)
    else:
        self.observation_space = spaces.Box(low=self.x_limits[:,0], high=self.x_limits[:,1], dtype=np.float32)

  def step(self, action):
    # If scaling actions!
    if (self.normalized_actions):
        action = 0.5*((self.u_limits[:,0] + self.u_limits[:,1]) + action*(self.u_limits[:,1] - self.u_limits[:,0]))
    a = action[:,np.newaxis]
    x = self.state[:,np.newaxis]
    cost = sum((x - self.goal) * np.matmul(self.Q, x - self.goal)) + sum((a - self.u0) * np.matmul(self.R, a - self.u0))
    cost = cost * self.dt
    reward = -cost[0]

    self.state = x + self.dt*(np.matmul(self.A, x) + np.matmul(self.B, a))
    # limit_violation = (observation > self.x_limits[:,1]).any() or (observation < self.x_limits[:,0]).any()
    self.state = np.minimum(1.2*self.x_limits[:,1:2], np.maximum(1.2*self.x_limits[:,0:1], self.state))
    reached_goal = np.linalg.norm(self.state - self.goal) < 5e-3
    if (self.add_quad_feat):
        observation = self.add_quadratic_features(self.state)
    else:
        observation = self.state

    self.state = self.state[:,0]
    observation = observation[:,0]
    self.step_count += 1

    # if (limit_violation):
    #   reward -= 100

    if ((self.step_count >= self.horizon) or reached_goal):
        done = True
        info = {'terminal_state': self.state, 'step_count' : self.step_count}
    else:
        done = False
        info = {'terminal_state': np.array([]), 'step_count' : self.step_count}

    return np.float32(observation), reward, done, info

  def reset(self, state=None):
    self.step_count = 0
    if (state is None):
      self.state = self.x_limits[:,0] + np.random.rand(self.X_DIMS) * (self.x_limits[:,1] - self.x_limits[:,0])
    else:
      self.state = state

    if (self.add_quad_feat):
        observation = self.add_quadratic_features(self.state[:,np.newaxis])
        observation = observation[:,0]
    else:
        observation = self.state

    return np.float32(observation)  # reward, done, info can't be included

  def add_quadratic_features(self, x):
    z = np.zeros((int(1/2*self.X_DIMS*(self.X_DIMS + 3)), x.shape[1]))
    z[0:self.X_DIMS, :] = x
    quad_terms = np.matmul(np.reshape(x.T, (x.shape[1], self.X_DIMS, 1)), np.reshape(x.T, (x.shape[1], 1, self.X_DIMS)))
    quad_terms_select = np.repeat(np.reshape(np.triu(np.ones((self.X_DIMS, self.X_DIMS))==1, 0), (1,self.X_DIMS,self.X_DIMS)), x.shape[1], axis=0)
    z[self.X_DIMS:, :] = np.reshape(quad_terms[quad_terms_select], (x.shape[1], int(1/2*self.X_DIMS*(self.X_DIMS + 1)))).T

    return z
    
  def render(self, mode='human'):
    pass

  def close (self):
    pass