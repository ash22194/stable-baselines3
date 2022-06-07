
import torch
from torch import nn
import os
import json
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.io import loadmat, savemat
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
	   'T': 12, 'dt': 2.5e-3, 'lambda_': 3, 'X_DIMS': 6, 'U_DIMS': 4,\
	   'x_limits': np.array([[l0 - 0.6, l0 + 0.4], [np.pi/2 - 0.3, np.pi/2 + 0.6], [-3, 3], [-3, 3], [-np.pi/2, np.pi/2], [-4, 4]]),\
	   'u_limits': np.array([[0, 3*m*g], [0, 3*m*g], [-0.25*m*g, 0.25*m*g], [-0.25*m*g, 0.25*m*g]])}
# sys = {'m': m, 'I': 3, 'l0': 1.15, 'd': 0.2, 'df': 0.5, 'g': g,\
# 	   'Q': np.diag([350, 700, 1.5, 1.5, 500, 5]), 'R': np.diag([0.000001, 0.000001, 0.00001, 0.00001]),\
# 	   'goal': np.array([[lg], [alpha1g], [0], [0], [0], [0]]),\
# 	   'u0': np.array([[m*g*np.cos(alpha2g)/np.sin(alpha1g - alpha2g)], [-m*g*np.cos(alpha1g)/np.sin(alpha1g - alpha2g)], [0], [0]]),\
# 	   'T': 4, 'dt': 2e-3, 'lambda_': 3, 'X_DIMS': 6, 'U_DIMS': 4,\
# 	   'x_limits': np.array([[l0 - 0.6, l0 + 0.1], [np.pi/2, np.pi/2 + 0.6], [-1, 1], [-2, 2], [-np.pi/8, np.pi/8], [-2.5, 2.5]]),\
# 	   'u_limits': np.array([[0, 3*m*g], [0, 3*m*g], [-0.25*m*g, 0.25*m*g], [-0.25*m*g, 0.25*m*g]])}
sys['gamma_'] = 1 - sys['lambda_'] * sys['dt']
fixed_start = False
normalized_actions = True
run_tests = False
env = Biped2D(sys, fixed_start=fixed_start, normalized_actions=normalized_actions)
check_env(env)

# Compute Policy and Value function numerically
algorithm = 'PPO'
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

total_timesteps = 18000000
model_load = False
if ((save_timestep <= total_timesteps) and (save_timestep > 0)):
	if (algorithm == 'A2C'):
		model = A2C.load(os.path.join(save_path, 'model_'+str(save_timestep)))
	elif (algorithm == 'PPO'):
		model = PPO.load(os.path.join(save_path, 'model_'+str(save_timestep)))
	elif (algorithm == 'DDPG'):
		model = DDPG.load(os.path.join(save_path, 'model_'+str(save_timestep)))
	model.set_env(env)
	model_load = True
else:
	policy_std = 0.1
	if (not normalized_actions):
		policy_std = policy_std * sys['u_limits'][:,1]
	n_steps = 100

	if (algorithm == 'A2C'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[128, 128], vf=[128, 128])], log_std_init=np.log(policy_std), optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))
		model = A2C('MlpPolicy', env, gamma=sys['gamma_'], n_steps=n_steps, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
	elif (algorithm == 'PPO'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[128, 128], vf=[128, 128])], log_std_init=np.log(policy_std), optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))
		model = PPO('MlpPolicy', env, gamma=sys['gamma_'], n_steps=n_steps, n_epochs=1, batch_size=n_steps, clip_range_vf=None, clip_range=0.2, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
	elif (algorithm == 'DDPG'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=dict(pi=[16, 16], qf=[16, 16]))
		model = DDPG('MlpPolicy', env, gamma=sys['gamma_'], tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)

	with open(os.path.join(save_path, 'parameters.txt'), 'w') as param_file:
		param = deepcopy(sys)
		param.update(policy_kwargs)

		param.update({'total_timesteps': total_timesteps})
		param.update({'n_steps': n_steps})
		for key in param.keys():
			if (type(param[key]).__module__ is np.__name__):
				param[key] = param[key].tolist()
			elif (type(param[key]) is type):
				param[key] = param[key].__name__

		param_file.write(json.dumps(param))

# save_every = total_timesteps - save_timestep
save_every = 2000000
timesteps = save_timestep
log_steps = 4000
while timesteps < total_timesteps:
	model.learn(total_timesteps=save_every, log_interval=round(log_steps/model.n_steps), reset_num_timesteps=(not model_load))
	timesteps = timesteps + save_every
	model.save(os.path.join(save_path, 'model_' + str(timesteps)))
	model_load = True

if (not run_tests):
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
	ax1 = fig.add_subplot(311)
	ax1.set_xlabel('l')
	ax1.set_ylabel('alpha')
	for t in range(num_trajectories):
		plt.plot(trajectories[0, :, t], trajectories[1, :, t], colors[t])

	ax2 = fig.add_subplot(312)
	ax2.set_xlabel('vx')
	ax2.set_ylabel('vz')
	for t in range(num_trajectories):
		plt.plot(trajectories[2, :, t], trajectories[3, :, t], colors[t])

	ax3 = fig.add_subplot(313)
	ax3.set_xlabel('theta')
	ax3.set_ylabel('theta-dot')
	for t in range(num_trajectories):
		plt.plot(trajectories[4, :, t], trajectories[5, :, t], colors[t])

	plt.show()

else:
	if (os.path.isfile(os.path.join(save_path, 'test_starts.mat'))):
		contents = loadmat(os.path.join(save_path, 'test_starts.mat'))
		starts = contents.get('starts')
	else:
		num_starts = 50
		starts = np.zeros((sys['X_DIMS'], num_starts))
		for s in range(num_starts):
			starts[:,s] = env.reset()

		savemat(os.path.join(save_path, 'test_starts.mat'), {'starts' : starts})

	if (starts is not None):
		num_starts = starts.shape[1]
		state_trajectories = np.zeros((sys['X_DIMS'], int(sys['T']/sys['dt'])+1, num_starts))
		action_trajectories = np.zeros((sys['U_DIMS'], int(sys['T']/sys['dt']), num_starts))
		value_estimates = np.zeros((num_starts, 1))

		for s in range(num_starts):
			obs = env.reset(state=starts[:,s])
			state_trajectories[:,0,s] = obs
			start = obs
			discount = 1
			for i in range(int(sys['T']/sys['dt'])):
				action, _state = model.predict(obs, deterministic=True)
				obs, reward, done, info = env.step(action)
				if (normalized_actions):
					action = 0.5*((sys['u_limits'][:,0] + sys['u_limits'][:,1]) + action*(sys['u_limits'][:,1] - sys['u_limits'][:,0]))

				state_trajectories[:,i+1,s] = obs
				action_trajectories[:,i,s] = action
				value_estimates[s,0] += discount * reward
				discount = discount * sys['gamma_']
				if done:
					print('Start state :', start, ', Final state :', obs)
					break

		savemat(os.path.join(save_path, 'test_trajectories.mat'), {'state_trajectories' : state_trajectories,
																   'action_trajectories' : action_trajectories,
																   'value_estimates' : value_estimates})


