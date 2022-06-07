
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

from systems.cartpole import CartPole
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

sys = {'mc': 5, 'mp': 1, 'l': 0.9, 'g': 9.81, 'Q': np.diag([25, 0.02, 25, 0.02]), 'R': np.diag([0.001, 0.001]),\
	   'goal': np.array([[0], [0], [np.pi], [0]]), 'u0': np.zeros((2,1)), 'T': 4, 'dt': 2e-3, 'lambda_': 3, 'X_DIMS': 4, 'U_DIMS': 2,\
	   'x_limits': np.array([[-1.5, 1.5], [-3, 3], [0, 2*np.pi], [-3, 3]]), 'u_limits': np.array([[-9, 9], [-9, 9]])}
sys['gamma_'] = np.exp(-sys['lambda_']*sys['dt'])

# sys = {'mc': 5, 'mp': 1, 'l': 0.9, 'g': 9.81, 'Q': np.diag([25, 0.02, 25, 0.02]), 'R': np.diag([0.001, 0.001]),\
# 	   'goal': np.array([[0], [0], [np.pi], [0]]), 'u0': np.zeros((2,1)), 'T': 4, 'dt': 1e-3, 'gamma_': 0.997, 'X_DIMS': 4, 'U_DIMS': 2,\
# 	   'x_limits': np.array([[-1, 1], [-1, 1], [3*np.pi/4, 5*np.pi/4], [-1, 1]]), 'u_limits': np.array([[-9, 9], [-9, 9]])}
fixed_start = False
normalized_actions = True
run_tests = False
env = CartPole(sys, fixed_start=fixed_start, normalized_actions=normalized_actions)
check_env(env)

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

# Compute Policy and Value function numerically
algorithm = 'PPO'
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
	n_steps = 50

	if (algorithm == 'A2C'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[32, 32], vf=[32, 32])], log_std_init=np.log(policy_std), optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))
		model = A2C('MlpPolicy', env, gamma=sys['gamma_'], n_steps=n_steps, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
	elif (algorithm == 'PPO'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[64, 64], vf=[64, 64])], log_std_init=np.log(policy_std), optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))
		model = PPO('MlpPolicy', env, gamma=sys['gamma_'], n_steps=n_steps, n_epochs=1, batch_size=n_steps, clip_range_vf=None, clip_range=0.2, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
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
ax1 = fig.add_subplot(211)
ax1.set_xlabel('x')
ax1.set_ylabel('x-dot')
for t in range(num_trajectories):
	plt.plot(trajectories[0, :, t], trajectories[1, :, t], colors[t])

ax2 = fig.add_subplot(212)
ax2.set_xlabel('th')
ax2.set_ylabel('th-dot')
for t in range(num_trajectories):
	plt.plot(trajectories[2, :, t], trajectories[3, :, t], colors[t])

plt.show()

if (run_tests):
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
