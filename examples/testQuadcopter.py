
import torch
from torch import nn
import os

import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace
from scipy.linalg import solve_continuous_are
from scipy.io import loadmat
from scipy.interpolate import interpn

from systems.quadcopter import Quadcopter
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

m = 0.5
g = 9.81
# sys = {'m': m, 'I': np.diag([4.86*1e-3, 4.86*1e-3, 8.8*1e-3]), 'l': 0.225, 'g': g, 'bk': 1.14*1e-7/(2.98*1e-6),\
# 	   'Q': np.diag([5, 0.001, 0.001, 5, 0.5, 0.5, 0.05, 0.075, 0.075, 0.05]), 'R': np.diag([0.002, 0.01, 0.01, 0.004]),\
# 	   'goal': np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]), 'u0': np.array([[m*g], [0], [0], [0]]),\
# 	   'T': 4, 'dt': 2.5e-3, 'gamma_': 0.99975, 'X_DIMS': 10, 'U_DIMS': 4,\
# 	   'x_limits': np.array([[0.0, 1.5], [-np.pi/2, np.pi/2], [-np.pi/2, np.pi/2], [-np.pi, np.pi], [-2, 2], [-2, 2], [-1.5, 1.5], [-6, 6], [-6, 6], [-2.5, 2.5]]),\
# 	   'u_limits': np.array([[0, 2*m*g], [-0.25*m*g, 0.25*m*g], [-0.25*m*g, 0.25*m*g], [-0.125*m*g, 0.125*m*g]])}
sys = {'m': m, 'I': np.diag([4.86*1e-3, 4.86*1e-3, 8.8*1e-3]), 'l': 0.225, 'g': g, 'bk': 1.14*1e-7/(2.98*1e-6),\
	   'Q': np.diag([5, 0.001, 0.001, 5, 0.5, 0.5, 0.05, 0.075, 0.075, 0.05]), 'R': np.diag([0.002, 0.01, 0.01, 0.004]),\
	   'goal': np.array([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]), 'u0': np.array([[m*g], [0], [0], [0]]),\
	   'T': 4, 'dt': 1e-3, 'gamma_': 0.99975, 'X_DIMS': 10, 'U_DIMS': 4,\
	   'x_limits': np.array([[-1.0, 3.0], [-np.pi/2, np.pi/2], [-np.pi/2, np.pi/2], [-np.pi, np.pi], [-4, 4], [-4, 4], [-4, 4], [-12, 12], [-12, 12], [-6, 6]]),\
	   'u_limits': np.array([[0, 2*m*g], [-0.25*m*g, 0.25*m*g], [-0.25*m*g, 0.25*m*g], [-0.125*m*g, 0.125*m*g]])}

fixed_start = False
normalized_actions = True
env = Quadcopter(sys, fixed_start=fixed_start, normalized_actions=normalized_actions)
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
	save_path = os.path.join('examples/data/quadcopter_fixedstart', algorithm)
else:
	save_path = os.path.join('examples/data/quadcopter', algorithm)

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

total_timesteps = 20000000
if ((save_timestep <= total_timesteps) and (save_timestep > 0)):
	if (algorithm == 'A2C'):
		model = A2C.load(os.path.join(save_path, 'model_'+str(save_timestep)))
	elif (algorithm == 'PPO'):
		model = PPO.load(os.path.join(save_path, 'model_'+str(save_timestep)))
	elif (algorithm == 'DDPG'):
		model = DDPG.load(os.path.join(save_path, 'model_'+str(save_timestep)))
	model.set_env(env)
else:
	if (normalized_actions):
		policy_std = 0.1
	else:
		policy_std = 0.1 * sys['u_limits'][:,1]

	if (algorithm == 'A2C'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[128, 128, 128, 128], vf=[128, 128, 128, 128])], log_std_init=policy_std, optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))
		model = A2C('MlpPolicy', env, gamma=sys['gamma_'], n_steps=40, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
	elif (algorithm == 'PPO'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[32, 32], vf=[32, 32])])
		model = PPO('MlpPolicy', env, gamma=sys['gamma_'], n_steps=env.horizon, clip_range_vf=None, clip_range=0.5, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
	elif (algorithm == 'DDPG'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=dict(pi=[16, 16], qf=[16, 16]))
		model = DDPG('MlpPolicy', env, gamma=sys['gamma_'], tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)

save_every = total_timesteps - save_timestep
timesteps = save_timestep
log_steps = 4000
while timesteps < total_timesteps:
	model.learn(total_timesteps=save_every, log_interval=round(log_steps/model.n_steps))
	timesteps = timesteps + save_every
	model.save(os.path.join(save_path, 'model_' + str(timesteps)))

# Test the learned policy
num_trajectories = 4
trajectories = np.zeros((sys['X_DIMS'], int(sys['T']/sys['dt']), num_trajectories))
for t in range(num_trajectories):
	obs = env.reset()
	start = obs
	for i in range(int(sys['T']/sys['dt'])):
		action, _state = model.predict(obs, deterministic=True)
		obs, reward, done, info = env.step(action)
		trajectories[:,i,t] = obs
		if done:
			print('Start state :', start, ', Final state :', obs)
			break

fig = plt.figure()
colors = ['r', 'g', 'b', 'm']
ax1 = fig.add_subplot(511)
ax1.set_xlabel('z')
ax1.set_ylabel('z-dot')
for t in range(num_trajectories):
	plt.plot(trajectories[0, :, t], trajectories[6, :, t], colors[t])

ax2 = fig.add_subplot(512)
ax2.set_xlabel('roll')
ax2.set_ylabel('roll-dot')
for t in range(num_trajectories):
	plt.plot(trajectories[1, :, t], trajectories[7, :, t], colors[t])

ax3 = fig.add_subplot(513)
ax3.set_xlabel('pitch')
ax3.set_ylabel('pitch-dot')
for t in range(num_trajectories):
	plt.plot(trajectories[2, :, t], trajectories[8, :, t], colors[t])

ax4 = fig.add_subplot(514)
ax4.set_xlabel('yaw')
ax4.set_ylabel('yaw-dot')
for t in range(num_trajectories):
	plt.plot(trajectories[3, :, t], trajectories[9, :, t], colors[t])

ax5 = fig.add_subplot(515)
ax5.set_xlabel('x-dot')
ax5.set_ylabel('y-dot')
for t in range(num_trajectories):
	plt.plot(trajectories[4, :, t], trajectories[5, :, t], colors[t])

plt.show()
set_trace()
