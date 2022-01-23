
import torch
from torch import nn
import os

import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace
from scipy.linalg import solve_continuous_are
from scipy.io import loadmat
from scipy.interpolate import interpn

from systems.biped2d import Biped2D
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

m = 72
g = 9.81
l0 = 1.15
df = 0.5
lg = 0.96
alpha1g = np.pi/2 + np.arcsin(df/2/lg)
l2g = np.sqrt((df + lg*np.cos(alpha1g))**2 + (lg*np.sin(alpha1g))**2)
alpha2g = np.arccos((df + lg*np.cos(alpha1g))/l2g)

sys = {'m': m, 'I': 3, 'l0': 1.15, 'd': 0.2, 'df': 0.5, 'g': g,\
	   'Q': np.diag([350, 700, 1.5, 1.5, 500, 5]), 'R': np.diag([0.000001, 0.000001, 0.00001, 0.00001]),\
	   'goal': np.array([[lg], [alpha1g], [0], [0], [0], [0]]),\
	   'u0': np.array([[m*g*np.cos(alpha2g)/np.sin(alpha1g - alpha2g)], [-m*g*np.cos(alpha1g)/np.sin(alpha1g - alpha2g)], [0], [0]]),\
	   'T': 4, 'dt': 2.5e-3, 'gamma_': 0.997, 'X_DIMS': 6, 'U_DIMS': 4,\
	   'x_limits': np.array([[l0 - 0.3, l0 + 0.1], [np.pi/2, np.pi/2 + 0.6], [-0.3, 0.5], [-0.5, 1], [-np.pi/8, np.pi/8], [-2, 2]]),\
	   'u_limits': np.array([[0, 3*m*g], [0, 3*m*g], [-0.25*m*g, 0.25*m*g], [-0.25*m*g, 0.25*m*g]])}
fixed_start = False
normalized_actions = True
env = Biped2D(sys, fixed_start=fixed_start, normalized_actions=normalized_actions)
check_env(env)

# Load DP solution to compare
# filename = '~/Documents/MATLAB/iLQG/DP/data/quadcopter/decomp0/final.mat'
# policy_analytical = loadmat(filename)

# Test the policy
# obs = env.reset()
# start = obs
# for i in range(12000):
#     action = interpn()
#     obs, reward, done, info = env.step(action)
#     if done:
#       print('Start state :', start, ', Final state :', obs)
#       obs = env.reset()
#       start = obs

# Compute Policy and Value function numerically
algorithm = 'A2C'
if (fixed_start):
	save_path = os.path.join('examples/data/biped2d_fixedstart', algorithm)
else:
	save_path = os.path.join('examples/data/biped2d', algorithm)
log_path = os.path.join(save_path, 'tb_log')
files = [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
save_timestep = 0
ff_latest = ''
for ff in files:
	if 'model' not in ff:
		continue 
	tt = ff.split('_')[-1]
	tt = int(tt.split('.')[0])
	if (tt > save_timestep):
		save_timestep = tt
		ff_latest = ff

total_timesteps = 10000000
if ((save_timestep <= total_timesteps) and (save_timestep > 0)):
	if (algorithm == 'A2C'):
		model = A2C.load(os.path.join(save_path, 'model_'+str(save_timestep)))
	elif (algorithm == 'PPO'):
		model = PPO.load(os.path.join(save_path, 'model_'+str(save_timestep)))
	elif (algorithm == 'DDPG'):
		model = DDPG.load(os.path.join(save_path, 'model_'+str(save_timestep)))
else:
	if (normalized_actions):
		policy_std = 0.1
	else:
		policy_std = 0.1 * sys['u_limits'][:,1]

	if (algorithm == 'A2C'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[32, 32], vf=[32, 32])], log_std_init=np.log(policy_std), optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))
		model = A2C('MlpPolicy', env, gamma=sys['gamma_'], n_steps=50, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
	elif (algorithm == 'PPO'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[32, 32], vf=[32, 32])])
		model = PPO('MlpPolicy', env, gamma=sys['gamma_'], n_steps=2048, clip_range_vf=None, clip_range=0.5, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
	elif (algorithm == 'DDPG'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=dict(pi=[16, 16], qf=[16, 16]))
		model = DDPG('MlpPolicy', env, gamma=sys['gamma_'], tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)

save_every = total_timesteps
timesteps = save_timestep
log_steps = 4000
while timesteps < total_timesteps:
	model.learn(total_timesteps=save_every, log_interval=round(log_steps/model.n_steps))
	timesteps = timesteps + save_every
	model.save(os.path.join(save_path, 'model_' + str(timesteps)))

# Test the learned policy
obs = env.reset()
start = obs
for i in range(int(4*sys['T']/sys['dt'])):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
      print('Start state :', start, ', Final state :', obs)
      obs = env.reset()
      start = obs

set_trace()
