
import torch
from torch import nn
import os

import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace
from scipy.linalg import solve_continuous_are

from systems.linearsystem import LinearSystem
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

sys = {'A': np.array([[0, 1], [0, 0]]), 'B': np.array([[0], [1]]), 'Q': np.diag([5, 0.25]), 'R': np.array([[0.01]]),\
	   'goal': np.zeros((2,1)), 'u0': np.zeros((1,1)), 'T': 4, 'dt': 1e-3, 'gamma_': 0.997, 'X_DIMS': 2, 'U_DIMS': 1,\
	   'x_limits': np.array([[-1, 1], [-3, 3]]), 'u_limits': np.array([[-50, 50]])}
add_quad_feat = False
normalized_actions = True
env = LinearSystem(sys, add_quad_feat=add_quad_feat, normalized_actions=normalized_actions)
check_env(env)

num_points = np.array([51, 151])
x_pts = np.linspace(sys['x_limits'][0,0], sys['x_limits'][0,1], num_points[0])
x_dot_pts = np.linspace(sys['x_limits'][1,0], sys['x_limits'][1,1], num_points[1])

[gx, gx_dot] = np.meshgrid(x_pts, x_dot_pts)
garray = np.concatenate((np.reshape(gx, (1, num_points[0]*num_points[1])), \
                         np.reshape(gx_dot, (1, num_points[0]*num_points[1]))), axis=0)

# Compute analytical solution
lambda_ = (1 - sys['gamma_']) / sys['dt']
P_analytical = solve_continuous_are(sys['A'] - lambda_/2*np.eye(sys['X_DIMS']), sys['B'], sys['Q'], sys['R'])
V_analytical = np.sum(garray * np.matmul(P_analytical, garray), axis=0)
K_analytical = np.matmul(np.linalg.inv(sys['R']), np.matmul(sys['B'].T, P_analytical))
policy_analytical = -np.matmul(K_analytical, garray)
policy_analytical = np.reshape(policy_analytical, num_points, order='F')

# Test the linear policy
obs = env.reset()
start = obs[0:sys['X_DIMS']]
for i in range(12000):
	action = np.matmul(-K_analytical, obs[0:sys['X_DIMS'],np.newaxis])[:,0]
	# If scaling actions - update the environment accordingly!
	if (normalized_actions):
		action = 2 * (action - 0.5*(sys['u_limits'][:,0] + sys['u_limits'][:,1])) / (sys['u_limits'][:,1] - sys['u_limits'][:,0])
	obs, reward, done, info = env.step(action)
	if done:
		print('Start state :', start, ', Final state :', obs[0:sys['X_DIMS']])
		obs = env.reset()
		start = obs[0:sys['X_DIMS']]

# Compute Policy and Value function numerically
algorithm = 'PPO'
save_path = os.path.join('examples/data/linear', algorithm)
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

total_timesteps = 5000000
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
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[8, 8], vf=[8, 8])], log_std_init=np.log(policy_std))
		model = A2C('MlpPolicy', env, gamma=sys['gamma_'], n_steps=500, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
	elif (algorithm == 'PPO'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[8, 8], vf=[8, 8])], log_std_init=np.log(policy_std), optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5))
		model = PPO('MlpPolicy', env, gamma=sys['gamma_'], n_steps=500, clip_range_vf=None, clip_range=0.2, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
	elif (algorithm == 'DDPG'):
		policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=dict(pi=[16, 16], qf=[16, 16]))
		model = DDPG('MlpPolicy', env, gamma=sys['gamma_'], train_freq=(1, "episode"), tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)

# save_every = total_timesteps
save_every = 500000
timesteps = save_timestep
log_steps = 4000
while timesteps < total_timesteps:
	if (algorithm=='A2C') or (algorithm=='PPO'):
		model.learn(total_timesteps=save_every, log_interval=round(log_steps/model.n_steps))
	elif (algorithm=='DDPG'):
		model.learn(total_timesteps=save_every, log_interval=1)
	timesteps = timesteps + save_every
	model.save(os.path.join(save_path, 'model_' + str(timesteps)))

if (add_quad_feat):
	policy_numerical = model.predict(env.add_quadratic_features(garray).T, deterministic=True)
else:
	policy_numerical = model.predict(garray.T, deterministic=True)
policy_numerical = policy_numerical[0]

if (normalized_actions):
	policy_numerical = 0.5*((sys['u_limits'][:,0] + sys['u_limits'][:,1]) + policy_numerical*(sys['u_limits'][:,1] - sys['u_limits'][:,0]))
policy_numerical = np.reshape(policy_numerical, num_points, order='F')

# Test the learned policy
obs = env.reset()
start = obs[0:sys['X_DIMS']]
for i in range(24000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
		print('Start state :', start, ', Final state :', obs[0:sys['X_DIMS']])
		obs = env.reset()
		start = obs[0:sys['X_DIMS']]

set_trace()
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_xlabel('x')
ax1.set_ylabel('x-dot')
ax1.set_title('CARE')
im1 = ax1.imshow(policy_analytical)
plt.colorbar(im1, ax=ax1)

ax2 = fig.add_subplot(212)
ax2.set_xlabel('x')
ax2.set_ylabel('x-dot')
ax2.set_title(algorithm)
im2 = ax2.imshow(policy_numerical, vmin=np.min(policy_analytical), vmax=np.max(policy_analytical))
plt.colorbar(im2, ax=ax2)

plt.show()
