
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

from typing import Callable
from systems.quadcopter import Quadcopter
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
	   'T': 4, 'dt': 2.5e-3, 'lambda_': 1, 'X_DIMS': 10, 'U_DIMS': 4,\
	   'x_limits': np.array([[0, 2.0], [-np.pi/2, np.pi/2], [-np.pi/2, np.pi/2], [-np.pi, np.pi], [-4, 4], [-4, 4], [-4, 4], [-3, 3], [-3, 3], [-3, 3]]),\
	   'u_limits': np.array([[0, 2*m*g], [-0.25*m*g, 0.25*m*g], [-0.25*m*g, 0.25*m*g], [-0.125*m*g, 0.125*m*g]])}
sys['gamma_'] = np.exp(-sys['lambda_']*sys['dt'])
# sys['I'][1,1] = 0.5*(sys['I'][0,0] + sys['I'][2,2])

fixed_start = False
normalized_actions = True
run_tests = True
env = Quadcopter(sys, fixed_start=fixed_start, normalized_actions=normalized_actions)
check_env(env)

# Compute Policy and Value function numerically
algorithm = 'A2C'
if (fixed_start):
	directory = 'examples/data/quadcopter_fixedstart'
else:
	directory = 'examples/data/quadcopter'
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
	n_steps = 256

	if (algorithm == 'A2C'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[2700, 2700, 2700], vf=[2700, 2700, 2700])], log_std_init=policy_std)
		model = A2C('MlpPolicy', env, learning_rate=linear_schedule(0.0015), use_rms_prop=True, gamma=sys['gamma_'], #gae_lambda=0.99, ent_coef=1.44e-8, vf_coef=0.993,
					n_steps=n_steps, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
	elif (algorithm == 'PPO'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[256, 256], vf=[256, 256])], log_std_init=policy_std)
		model = PPO('MlpPolicy', env, learning_rate=linear_schedule(0.0005), gamma=sys['gamma_'], n_steps=n_steps,
			        n_epochs=1, batch_size=n_steps, clip_range_vf=None, clip_range=0.2, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
	elif (algorithm == 'DDPG'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=dict(pi=[16, 16], qf=[16, 16]))
		model = DDPG('MlpPolicy', env, gamma=sys['gamma_'], tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)

	with open(os.path.join(save_path, 'parameters.txt'), 'w') as param_file:
		param = deepcopy(sys)
		param.update(policy_kwargs)

		param.update({'total_timesteps': total_timesteps})
		param.update({'n_steps' : n_steps})
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
	trajectories_full = np.zeros((sys['X_DIMS']+2, int(sys['T']/sys['dt']), num_trajectories))
	for t in range(num_trajectories):
		obs = env.reset()
		obs_full = np.concatenate([np.zeros((2,1)), obs[:,np.newaxis]])
		start = obs
		for i in range(int(sys['T']/sys['dt'])):
			action, _state = model.predict(obs, deterministic=True)
			obs, reward, done, info = env.step(action)
			if (normalized_actions):
				action = 0.5*((sys['u_limits'][:,0] + sys['u_limits'][:,1]) + action*(sys['u_limits'][:,1] - sys['u_limits'][:,0]))
			obs_full = env.dyn_full_rk4(obs_full, action[:,np.newaxis], sys['dt'])
			err = np.linalg.norm(obs_full[2:,0] - obs)
			trajectories[:,i,t] = obs
			trajectories_full[:,i,t] = obs_full[:,0]
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

else:
	if (os.path.isfile(os.path.join(directory, 'trajectories_GA_MCTS_Heuristic_Pareto.mat'))):
		contents = loadmat(os.path.join(directory, 'trajectories_GA_MCTS_Heuristic_Pareto.mat'))
		starts = contents.get('starts')
	else:
		num_starts = 50
		starts = np.zeros((sys['X_DIMS'], num_starts))
		for s in range(num_starts):
			starts[:,s] = env.reset()

		savemat(os.path.join(directory, 'test_starts.mat'), {'starts' : starts})

	# starts = np.array([[0.7000], [-0.3927], [-0.3927], [-0.7854], [-1.0000], [1.0000], [-0.7500], [-0.5000], [-0.5000], [-0.2500]])

	if (starts is not None):
		num_starts = starts.shape[1]
		state_trajectories = np.zeros((sys['X_DIMS'], int(sys['T']/sys['dt'])+1, num_starts))
		state_trajectories_full = np.zeros((sys['X_DIMS']+2, int(sys['T']/sys['dt'])+1, num_starts))
		action_trajectories = np.zeros((sys['U_DIMS'], int(sys['T']/sys['dt']), num_starts))
		value_estimates = np.zeros((num_starts, 1))

		for s in range(num_starts):
			obs = env.reset(state=starts[:,s])
			state_trajectories[:,0,s] = obs

			obs_full = np.concatenate([np.zeros((2,1)), obs[:,np.newaxis]])
			state_trajectories_full[:,0:1,s] = obs_full

			start = obs
			discount = 1
			for i in range(int(sys['T']/sys['dt'])):
				action, _state = model.predict(obs, deterministic=True)
				obs, reward, done, info = env.step(action)
				if (normalized_actions):
					action = 0.5*((sys['u_limits'][:,0] + sys['u_limits'][:,1]) + action*(sys['u_limits'][:,1] - sys['u_limits'][:,0]))
				obs_full = env.dyn_full_rk4(obs_full, action[:,np.newaxis], sys['dt'])

				state_trajectories[:,i+1,s] = obs
				state_trajectories_full[:,(i+1):(i+2),s] = obs_full
				obs_full = np.concatenate([obs_full[0:2,:], obs[:,np.newaxis]])
				action_trajectories[:,i,s] = action
				value_estimates[s,0] += discount * reward
				discount = discount * sys['gamma_']

				if done:
					print('Start state :', start, ', Final state :', obs_full[:,0])
					break
		savemat(os.path.join(directory, 'test_trajectories_' + algorithm + '.mat'), {'state_trajectories' : state_trajectories,
																   'state_trajectories_full' : state_trajectories_full,
																   'action_trajectories' : action_trajectories,
																   'value_estimates' : value_estimates})
		# savemat(os.path.join(save_path, 'test_trajectory.mat'), {'state_trajectories' : state_trajectories,
		# 														   'state_trajectories_full' : state_trajectories_full,
		# 														   'action_trajectories' : action_trajectories,
		# 														   'value_estimates' : value_estimates})