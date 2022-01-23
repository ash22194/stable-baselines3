
import torch
from torch import nn
import os

import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace
from scipy.linalg import solve_continuous_are
from scipy.io import loadmat
from scipy.interpolate import interpn

from systems.cartpole import CartPole
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

sys = {'mc': 5, 'mp': 1, 'l': 0.9, 'g': 9.81, 'Q': np.diag([25, 0.02, 25, 0.02]), 'R': np.diag([0.001, 0.001]),\
	   'goal': np.array([[0], [0], [np.pi], [0]]), 'u0': np.zeros((2,1)), 'T': 4, 'dt': 2.5e-3, 'gamma_': 0.997, 'X_DIMS': 4, 'U_DIMS': 2,\
	   'x_limits': np.array([[-1.5, 1.5], [-3, 3], [0, 2*np.pi], [-3, 3]]), 'u_limits': np.array([[-9, 9], [-9, 9]])}
# sys = {'mc': 5, 'mp': 1, 'l': 0.9, 'g': 9.81, 'Q': np.diag([25, 0.02, 25, 0.02]), 'R': np.diag([0.001, 0.001]),\
# 	   'goal': np.array([[0], [0], [np.pi], [0]]), 'u0': np.zeros((2,1)), 'T': 4, 'dt': 1e-3, 'gamma_': 0.997, 'X_DIMS': 4, 'U_DIMS': 2,\
# 	   'x_limits': np.array([[-1, 1], [-1, 1], [3*np.pi/4, 5*np.pi/4], [-1, 1]]), 'u_limits': np.array([[-9, 9], [-9, 9]])}
fixed_start = False
normalized_actions = True
env = CartPole(sys, fixed_start=fixed_start, normalized_actions=normalized_actions)
check_env(env)

# Load DP solution to compare
# filename = '~/Documents/MATLAB/iLQG/DP/data/cartpole/decomp0/final.mat'
# policy_analytical = loadmat(filename)

env_dummy = CartPole(sys, fixed_start=fixed_start, normalized_actions=normalized_actions)
check_env(env_dummy)
goal = sys['goal']
u0 = sys['u0']
A = np.zeros((sys['X_DIMS'], sys['X_DIMS']))
B = np.zeros((sys['X_DIMS'], sys['U_DIMS']))
for xx in range(sys['X_DIMS']):
	perturb_p = np.zeros(goal.shape)
	perturb_p[xx] = 1e-4
	perturb_m = np.zeros(goal.shape)
	perturb_m[xx] = -1e-4
	dyn_p = env_dummy.dyn(goal + perturb_p, u0)
	dyn_m = env_dummy.dyn(goal + perturb_m, u0)
	A[:, xx:(xx+1)] = (dyn_p - dyn_m) / (2e-4)

for uu in range(sys['U_DIMS']):
	perturb_p = np.zeros(u0.shape)
	perturb_p[uu] = 1e-4
	perturb_m = np.zeros(u0.shape)
	perturb_m[uu] = -1e-4
	dyn_p = env_dummy.dyn(goal, u0 + perturb_p)
	dyn_m = env_dummy.dyn(goal, u0 + perturb_m)
	B[:, uu:(uu+1)] = (dyn_p - dyn_m) / (2e-4)

Q = sys['Q']
R = sys['R']
lambda_ = (1 - sys['gamma_']) / sys['dt']
P = solve_continuous_are(A - lambda_/2*np.eye(sys['X_DIMS']), B, Q, R)
K = np.matmul(np.linalg.inv(R), np.matmul(B.T, P))

# Test the policy
obs = env_dummy.reset()
start = obs
u_limits = sys['u_limits']
value = 0
for i in range(int(3*sys['T']/sys['dt'])):
	action = np.matmul(-K, obs[:,np.newaxis] - goal)[:,0]
	action = np.maximum(u_limits[:,0], np.minimum(u_limits[:,1], action))
	if (normalized_actions):
		action = (2*action - (u_limits[:,1] + u_limits[:,0])) / (u_limits[:,1] - u_limits[:,0])

	obs, reward, done, info = env_dummy.step(action)
	value *= sys['gamma_']
	value += reward
	if done:
		print('Start state :', start, ', Final state :', obs, 'Value :', value)
		obs = env_dummy.reset()
		start = obs
		value = 0

set_trace()
# Compute Policy and Value function numerically
algorithm = 'A2C'
if (fixed_start):
	save_path = os.path.join('examples/data/cartpole_fixedstart', algorithm)
else:
	save_path = os.path.join('examples/data/cartpole', algorithm)
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
# save_every = 500000
timesteps = save_timestep
log_steps = 4000
while timesteps < total_timesteps:
	model.learn(total_timesteps=save_every, log_interval=round(log_steps/model.n_steps))
	timesteps = timesteps + save_every
	model.save(os.path.join(save_path, 'model_' + str(timesteps)))

# Test the learned policy
obs = env.reset()
start = obs
for i in range(24000):
	action, _state = model.predict(obs, deterministic=True)
	obs, reward, done, info = env.step(action)
	# if (i%100==0):
	# 	print('State : ', obs, ', Action : ', action)
		# set_trace()
	if done:
		print('Start state :', start, ', Final state :', obs)
		obs = env.reset()
		start = obs

set_trace()
