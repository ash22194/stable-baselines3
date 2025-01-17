
import torch
from torch import nn
import os
import time
import json
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from scipy.io import loadmat, savemat
from scipy.interpolate import interpn

from typing import Callable
from systems.manipulator4dof import Manipulator4DOF
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

m = [5.4, 1.8, 0.6, 0.2]
l = [0.2, 0.5, 0.25, 0.125]
g = 9.81
sys = {'m': m, 'l': l, 'g': g,\
	   'Q': np.diag([4,4,4,4,0.1,0.1,0.1,0.1]), 'R': np.diag([0.002,0.004,0.024,0.1440]),\
	   'goal': np.array([[np.pi],[0],[0],[0],[0],[0],[0],[0]]), 'u0': np.array([[0],[0],[0],[0]]),\
	   'T': 4, 'dt': 1e-3, 'lambda_': 3, 'X_DIMS': 8, 'U_DIMS': 4,\
	   'x_limits': np.array([[0, 2*np.pi],[-np.pi, np.pi],[-np.pi, np.pi],[-np.pi, np.pi],[-6, 6],[-6, 6],[-6, 6],[-6, 6]]),\
	   'u_limits': np.array([[-24, 24], [-15, 15], [-7.5, 7.5], [-1, 1]])}
sys['gamma_'] = np.exp(-sys['lambda_']*sys['dt'])

fixed_start = False
normalized_actions = True
run_tests = True
env = Manipulator4DOF(sys, fixed_start=fixed_start, normalized_actions=normalized_actions)
check_env(env)

# Compute Policy and Value function numerically
algorithm = 'PPO'
if (fixed_start):
	directory = 'examples/data/manipulator4dof_fixedstart'
else:
	directory = 'examples/data/manipulator4dof'
save_path = os.path.join(directory, algorithm)
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
	n_steps = 128

	if (algorithm == 'A2C'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[256, 256], vf=[256, 256])], log_std_init=policy_std)
		model = A2C('MlpPolicy', env, learning_rate=linear_schedule(0.00075), use_rms_prop=True, gamma=sys['gamma_'], 
					n_steps=n_steps, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
	elif (algorithm == 'PPO'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[256, 256], vf=[256, 256])], log_std_init=policy_std)
		model = PPO('MlpPolicy', env, learning_rate=linear_schedule(0.00035), gamma=sys['gamma_'], n_steps=n_steps, 
					use_sde=False, n_epochs=1, batch_size=n_steps, clip_range_vf=None, clip_range=0.2, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
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

else:
	if (os.path.isfile(os.path.join(directory, 'trajectories_GA_MCTS_Random_Pareto.mat'))):
		contents = loadmat(os.path.join(directory, 'trajectories_GA_MCTS_Random_Pareto.mat'))
		starts = contents.get('starts')
	else:
		num_starts = 50
		starts = np.zeros((sys['X_DIMS'], num_starts))
		for s in range(num_starts):
			starts[:,s] = env.reset()

		savemat(os.path.join(directory, 'test_starts.mat'), {'starts' : starts})

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
		savemat(os.path.join(directory, 'test_trajectories_' + algorithm + '.mat'), {'state_trajectories' : state_trajectories,
																   'action_trajectories' : action_trajectories,
																   'value_estimates' : value_estimates})
