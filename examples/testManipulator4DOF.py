
import torch
from torch import nn
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace
from scipy.linalg import solve_continuous_are
from scipy.io import loadmat
from scipy.interpolate import interpn

from systems.manipulator4dof import Manipulator4DOF
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

m = [5.4, 1.8, 0.6, 0.2]
l = [0.2, 0.5, 0.25, 0.125]
g = 9.81
sys = {'m': m, 'l': l, 'g': g,\
	   'Q': np.diag([4,4,4,4,0.1,0.1,0.1,0.1]), 'R': np.diag([0.002,0.004,0.024,0.1440]),\
	   'goal': np.array([[np.pi],[0],[0],[0],[0],[0],[0],[0]]), 'u0': np.array([[0],[0],[0],[0]]),\
	   'T': 4, 'dt': 1e-3, 'gamma_': 0.997, 'X_DIMS': 8, 'U_DIMS': 4,\
	   'x_limits': np.array([[0, 2*np.pi],[-np.pi, np.pi],[-np.pi, np.pi],[-np.pi, np.pi],[-6, 6],[-6, 6],[-6, 6],[-6, 6]]),\
	   'u_limits': np.array([[-24, 24], [-15, 15], [-7.5, 7.5], [-1, 1]])}

fixed_start = False
normalized_actions = True
env = Manipulator4DOF(sys, fixed_start=fixed_start, normalized_actions=normalized_actions)
check_env(env)

# Test dynamics
num_points = 100
states = np.zeros((sys['X_DIMS'], num_points))
inputs = np.zeros((sys['U_DIMS'], num_points))
d1 = np.zeros((sys['X_DIMS'], num_points))
d2 = np.zeros((sys['X_DIMS'], num_points))
d3 = np.zeros((sys['X_DIMS'], num_points))
t1 = 0
t2 = 0
t3 = 0
for nn in range(num_points):
	states[:,nn] = sys['x_limits'][:,0] + (sys['x_limits'][:,1] - sys['x_limits'][:,0])*np.random.rand(sys['X_DIMS'])
	inputs[:,nn] = sys['u_limits'][:,0] + (sys['u_limits'][:,1] - sys['u_limits'][:,0])*np.random.rand(sys['U_DIMS'])
	t1s = time.time()
	d1[:,nn:(nn+1)] = env.dyn(states[:,nn:(nn+1)], inputs[:,nn:(nn+1)])
	t1 += (time.time() - t1s)

	t2s = time.time()
	d2[:,nn:(nn+1)] = env.dyn_numerical(states[:,nn:(nn+1)], inputs[:,nn:(nn+1)])
	t2 += (time.time() - t2s)
	
	t3s = time.time()
	d3[:,nn:(nn+1)] = env.dyn_simscape(states[:,nn:(nn+1)], inputs[:,nn:(nn+1)])
	t3 += (time.time() - t3s)

set_trace()
# Load DP solution to compare
# filename = '~/Documents/MATLAB/iLQG/DP/data/manipulator4dof/decomp0/final.mat'
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
	save_path = os.path.join('examples/data/manipulator4dof_fixedstart', algorithm)
else:
	save_path = os.path.join('examples/data/manipulator4dof', algorithm)

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

total_timesteps = 40000000
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
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[64, 64, 64], vf=[64, 64, 64])], log_std_init=policy_std, optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))
		model = A2C('MlpPolicy', env, gamma=sys['gamma_'], n_steps=50, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
	elif (algorithm == 'PPO'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[32, 32], vf=[32, 32])])
		model = PPO('MlpPolicy', env, gamma=sys['gamma_'], n_steps=env.horizon, clip_range_vf=None, clip_range=0.5, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
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
ax1 = fig.add_subplot(411)
ax1.set_xlabel('th1')
ax1.set_ylabel('th1-dot')
for t in range(num_trajectories):
	plt.plot(trajectories[0, :, t], trajectories[4, :, t], colors[t])

ax2 = fig.add_subplot(412)
ax2.set_xlabel('th2')
ax2.set_ylabel('th2-dot')
for t in range(num_trajectories):
	plt.plot(trajectories[1, :, t], trajectories[5, :, t], colors[t])

ax3 = fig.add_subplot(413)
ax3.set_xlabel('th3')
ax3.set_ylabel('th3-dot')
for t in range(num_trajectories):
	plt.plot(trajectories[2, :, t], trajectories[6, :, t], colors[t])

ax4 = fig.add_subplot(414)
ax4.set_xlabel('th4')
ax4.set_ylabel('th4-dot')
for t in range(num_trajectories):
	plt.plot(trajectories[3, :, t], trajectories[7, :, t], colors[t])

plt.show()
set_trace()
