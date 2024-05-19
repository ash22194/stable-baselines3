import os
import argparse
import numpy as np
import torch as th
from copy import deepcopy
from ruamel.yaml import YAML
import gymnasium as gym

from stable_baselines3.gpu_systems import GPUQuadcopter, GPUQuadcopterDecomposition, GPUQuadcopterTT, GPUQuadcopterTTDecomposition

def parse_common_args(cfg: dict, env_device: str = 'cpu'):

	environment_args = cfg['environment']
	algorithm_args = cfg['algorithm']
	policy_args = cfg['policy']

	# setup algorithm parameters
	if ('algorithm_kwargs' not in algorithm_args.keys()):
		algorithm_args['algorithm_kwargs'] = dict()
	if ((env_device=='cuda') or (env_device=='mps')):
		algorithm_args['algorithm_kwargs']['device'] = env_device
	# check batch size
	batch_size = algorithm_args['algorithm_kwargs'].get('batch_size', None)
	if (batch_size is None):
		batch_size = 1
	if (batch_size <= 1):
		batch_size *= (algorithm_args['algorithm_kwargs']['n_steps']*environment_args['num_envs'])
	algorithm_args['algorithm_kwargs']['batch_size'] = int(batch_size)
	# setup the learning rate schedule
	learning_rate_schedule = algorithm_args['algorithm_kwargs'].pop('learning_rate_schedule', None)
	if (learning_rate_schedule is not None):
		learning_rate = algorithm_args['algorithm_kwargs'].pop('learning_rate', None)
		assert learning_rate is not None, 'learning rate schedule specified without learning rate'
		if (learning_rate_schedule['type'] == 'lin'):
			learning_rate = linear_schedule(learning_rate)
		elif (learning_rate_schedule['type'] == 'exp'):
			learning_rate = exponential_schedule(learning_rate, learning_rate_schedule['decay_rate'])
		elif (learning_rate_schedule['type'] == 'sawt'):
			learning_rate = decay_sawtooth_schedule(learning_rate, learning_rate_schedule['sawtooth_width'])
		elif (learning_rate_schedule['type'] == 'kla'):
			if (learning_rate_schedule['target_kl']==None):
				clip_range = algorithm_args['algorithm_kwargs'].get('clip_range', 0.2)
				learning_rate_schedule['target_kl'] = (clip_range**2)/4
			algorithm_args['algorithm_kwargs']['target_kl'] = learning_rate_schedule['target_kl']
		algorithm_args['algorithm_kwargs']['learning_rate'] = learning_rate

	# setup policy parameters
	if ('policy_kwargs' not in policy_args.keys()):
		policy_args['policy_kwargs'] = dict()
	# activation function
	if ('activation_fn' in policy_args['policy_kwargs'].keys()):
		if (policy_args['policy_kwargs']['activation_fn'] == 'relu'):
			policy_args['policy_kwargs']['activation_fn'] = nn.ReLU
		elif (policy_args['policy_kwargs']['activation_fn'] == 'elu'):
			policy_args['policy_kwargs']['activation_fn'] = nn.ELU
		elif (policy_args['policy_kwargs']['activation_fn'] == 'tanh'):
			policy_args['policy_kwargs']['activation_fn'] = nn.Tanh
	# optimizer args
	if ('optimizer_class' in policy_args['policy_kwargs'].keys()):
		if (policy_args['policy_kwargs']['optimizer_class'] == 'rmsprop'):
			policy_args['policy_kwargs']['optimizer_class'] = torch.optim.RMSprop
			policy_args['policy_kwargs']['optimizer_kwargs'] = dict(alpha=0.99, eps=1e-5, weight_decay=0)
		elif (policy_args['policy_kwargs']['optimizer_class'] == 'rmsproptflike'):
			policy_args['policy_kwargs']['optimizer_class'] = RMSpropTFLike 
			policy_args['policy_kwargs']['optimizer_kwargs'] = dict(eps=1e-5)
		elif (policy_args['policy_kwargs']['optimizer_class'] == 'adam'):
			policy_args['policy_kwargs']['optimizer_class'] = torch.optim.Adam
			policy_args['policy_kwargs']['optimizer_kwargs'] = None

	cfg['environment'] = environment_args
	cfg['algorithm'] = algorithm_args
	cfg['policy'] = policy_args

	return cfg

def get_env(env_name: str, env_device: str, env_kwargs=dict()):

	if (env_name == 'GPUQuadcopter'):
		env = GPUQuadcopter(device=env_device, **env_kwargs)
		env_decomp = GPUQuadcopterDecomposition(device=env_device, **env_kwargs)

	elif (env_name == 'Quadcopter-v0'):
		env = gym.make(env_name, **env_kwargs)
		env_decomp = gym.make('QuadcopterDecomposition-v0', **env_kwargs)

	elif (env_name == 'GPUQuadcopterTT'):
		env = GPUQuadcopterTT(device=env_device, **env_kwargs)
		env_decomp = GPUQuadcopterTTDecomposition(device=env_device, **env_kwargs)

	elif (env_name == 'QuadcopterTT-v0'):
		env = gym.make(env_name, **env_kwargs)
		env_decomp = gym.make('QuadcopterTTDecomposition-v0', **env_kwargs)

	return env, env_decomp

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='quadcopter', help='environemnt to test')
	parser.add_argument('--device', type=str, default='cuda', help='cuda/cpu')

	normalized_observations = False
	args = parser.parse_args()
	env_name = args.env_name
	env_device = args.device
	device_dir = env_device
	if (env_device == 'mps'):
		device_dir = 'cuda'
	elif not ((env_device=='cuda') or (env_device=='cpu')):
		NotImplementedError

	cfg_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../examples/configs', env_name, device_dir, 'ppo', 'sweep_optimized_cfg.yaml')
	if (not os.path.isfile(cfg_abs_path)):
		cfg_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../examples/configs', env_name, device_dir, 'ppo', 'cfg.yaml')

	cfg_orig = YAML().load(open(cfg_abs_path, 'r'))
	# cfg = parse_common_args(deepcopy(cfg_orig))
	cfg = deepcopy(cfg_orig)
	if (cfg['environment']['environment_kwargs'].get('intermittent_starts', None) is not None):
		cfg['environment']['environment_kwargs']['intermittent_starts'] = False

	env, env_decomp = get_env(cfg['environment']['name'], env_device, cfg['environment'].get('environment_kwargs', dict()))
	
	# rollout both envs with random actions
	num_episodes = 2
	for ee in range(num_episodes):
		obs = env.reset()
		obs_decomp = env_decomp.reset(state=env.state)

		done = False
		while (not done):
			if (env_device=='cuda') or (env_device=='mps'):
				action = 0.5*(env.th_action_space_high + env.th_action_space_low) + (th.rand(env.action_space.shape[0], device=env_device) - 0.5)*(env.th_action_space_high - env.th_action_space_low)
				action = th.reshape(action, (1,-1))
			else:
				action = 0.5*(env.action_space.high + env.action_space.low) + (np.random.rand(env.action_space.shape[0]) - 0.5)*(env.action_space.high - env.action_space.low)

			obs, r, done, _, _ = env.step(action)
			obs_decomp, r_decomp, done_decomp, _, _ = env_decomp.step(action)

			print(
				'state error :', np.linalg.norm((env.state - env_decomp.state).tolist()),
				'obs error :', np.linalg.norm((obs - obs_decomp).tolist()),
				'rew error :', r - r_decomp
			)
		

if __name__=='__main__':
	main()