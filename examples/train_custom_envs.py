import torch
from torch import nn
import os
import argparse
from shutil import copyfile
from ruamel.yaml import YAML
from cProfile import Profile

import gymnasium as gym
import numpy as np
from typing import Callable, Dict

from stable_baselines3 import A2CwReg, PPO, TD3
from stable_baselines3.common.callbacks import CustomSaveLogCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.learning_schedules import linear_schedule, decay_sawtooth_schedule, exponential_schedule
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

def initialize_model(config_file_path: str, algorithm: str, save_dir: str, run: int = None):
	
	algorithm = algorithm.upper()
	if (run is not None):
		# load from a previous run
		run_path = os.path.join(save_dir, algorithm, 'tb_log', algorithm + '_', str(run))
		assert os.path.isdir(run_path), 'run directory does not exist!'
		config_file_path = os.path.join(run_path, 'cfg.yaml')

	# load the config file and extract hyperparameters
	cfg = YAML().load(open(config_file_path, 'r'))
	# initialize the policy args
	policy_args = cfg['policy']
	if ('policy_kwargs' not in policy_args.keys()):
		policy_args['policy_kwargs'] = dict()

	environment_args = cfg['environment']

	algorithm_args = cfg['algorithm']
	if ('algorithm_kwargs' not in algorithm_args.keys()):
		algorithm_args['algorithm_kwargs'] = dict()

	if ('activation_fn' in policy_args['policy_kwargs'].keys()):
		if (policy_args['policy_kwargs']['activation_fn'] == 'relu'):
			policy_args['policy_kwargs']['activation_fn'] = nn.ReLU
		elif (policy_args['policy_kwargs']['activation_fn'] == 'elu'):
			policy_args['policy_kwargs']['activation_fn'] = nn.ELU
		elif (policy_args['policy_kwargs']['activation_fn'] == 'tanh'):
			policy_args['policy_kwargs']['activation_fn'] = nn.Tanh

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

	# initialize the environment
	normalized_rewards = environment_args.get('normalized_rewards')
	num_envs = environment_args.get('num_envs')
	env = make_vec_env(
		environment_args.get('name'), n_envs=num_envs, 
		env_kwargs=environment_args.get('environment_kwargs', dict()),
		vec_env_cls=DummyVecEnv
	)
	if (normalized_rewards):
		env = VecNormalize(
			env,
			norm_obs=False,
			norm_reward=True,
			training=True,
			gamma=algorithm_args['algorithm_kwargs'].get('gamma', 0.99)
		)

	# initialize the agent
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
		algorithm_args['algorithm_kwargs']['learning_rate'] = learning_rate

	if (algorithm_args.get('name') is not None):
		assert algorithm == algorithm_args.get('name').upper(), 'config file is for %s but requested %s' %(algorithm_args.get('name').upper(), algorithm)

	model_uninitialized = True
	if (run is not None):
		# load a model if it exists
		model_save_prefix = policy_args.get('save_prefix')
		files = [f for f in os.listdir(run_path) if os.path.isfile(os.path.join(run_path, f))]
		save_timestep = 0
		ff_latest = ''
		for ff in files:
			if model_save_prefix not in ff:
				continue 
			tt = ff.split('_')[-1]
			tt = int(tt.split('.')[0])
			if (tt > save_timestep):
				save_timestep = tt
				ff_latest = ff

		if (save_timestep > 0):
			if (algorithm == 'A2C'):
				model = A2CwReg.load(os.path.join(save_path, model_save_prefix + '_' + str(save_timestep)))
			elif (algorithm == 'PPO'):
				model = PPO.load(os.path.join(save_path, model_save_prefix + '_' + str(save_timestep)))
			elif (algorithm == 'TD3'):
				model = TD3.load(os.path.join(save_path, model_save_prefix + '_' + str(save_timestep)))
			model.set_env(env)
			model_uninitialized = False

	if (model_uninitialized): 
		# new training or unsaved model from a previous run
		if (algorithm == 'A2C'):
			model = A2CwReg('MlpPolicy', env, **algorithm_args.get('algorithm_kwargs'), policy_kwargs=policy_args.get('policy_kwargs'))
		elif (algorithm == 'PPO'):
			model = PPO('MlpPolicy', env, **algorithm_args.get('algorithm_kwargs'), policy_kwargs=policy_args.get('policy_kwargs'))
		elif (algorithm == 'TD3'):
			model = TD3('MlpPolicy', env, **algorithm_args.get('algorithm_kwargs'), policy_kwargs=policy_args.get('policy_kwargs'))

		log_path = os.path.join(save_dir, algorithm, 'tb_log')
		logger = configure_logger(verbose=1, tensorboard_log=log_path, tb_log_name=algorithm, reset_num_timesteps=True)
		# copy config file 
		copyfile(config_file_path, os.path.join(logger.get_dir(), 'cfg.yaml'))
		env_name = os.path.basename(os.path.dirname(os.path.dirname(config_file_path)))
		class_file_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../stable_baselines3/systems', env_name + '.py')
		copyfile(class_file_abs_path, os.path.join(logger.get_dir(), env_name + '.py'))
		model.set_logger(logger)

	save_every_timestep = algorithm_args.get('save_every_timestep')
	if (save_every_timestep is None):
		save_every_timestep = algorithm_args.get('total_timesteps')
	callback = CustomSaveLogCallback(save_every_timestep=save_every_timestep, save_path=logger.get_dir(), save_prefix=policy_args.get('save_prefix'))

	return model, callback, cfg

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='linearsystem', help='environment name')
	parser.add_argument('--algorithm', type=str, default='ppo', help='algorithm to train')
	parser.add_argument('--run', type=int, default=None, help='run number')
	parser.add_argument('--profile', default=False, action='store_true', help='profile this run?')
	args = parser.parse_args()
	env_name = args.env_name
	algo = args.algorithm.lower()
	run_id = args.run
	profile_run = args.profile

	cfg_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', env_name, algo, 'sweep_optimized_cfg.yaml')
	if (not os.path.isfile(cfg_abs_path)):
		cfg_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', env_name, algo, 'cfg.yaml')
	save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', env_name)

	model, callback, cfg = initialize_model(cfg_abs_path, algo, save_dir, run_id)

	# train the model
	total_timesteps = cfg['algorithm']['total_timesteps']
	model_load = model.num_timesteps > 0
	log_interval = 1 # number of steps after which logging happens = (num_envs*n_steps_to_update)*log_interval
	if ((total_timesteps-model.num_timesteps) > 0):
		if (profile_run):
			prof = Profile()
			prof.enable()

		model.learn(
			total_timesteps=total_timesteps-model.num_timesteps,
			log_interval=log_interval,
			callback=callback,
			reset_num_timesteps=(not model_load)
		)

		if (profile_run):
			prof.disable()
			prof.create_stats()
			prof.dump_stats(os.path.join(model.logger.dir, 'run_profile'))

	# evaluate the model
	test_env = gym.make(cfg['environment']['name'], **cfg['environment'].get('environment_kwargs', dict()))
	for ee in range(5):
		obs, _ = test_env.reset()
		start = test_env.get_obs(normalized=False)
		done = False
		ep_reward = 0
		ep_discounted_reward = 0
		discount = 1
		while (not done):
			action, _state = model.predict(obs, deterministic=True)
			obs, reward, done, _, info = test_env.step(action)

			ep_reward += reward
			ep_discounted_reward += (discount*reward)
			discount *= model.gamma

		end = test_env.get_obs(normalized=False)
		with np.printoptions(precision=3, suppress=True):
			print('start obs :', start)
			print('final obs :', end)
			print('reward (discounted) : %f (%f)' %(ep_reward, ep_discounted_reward))

if __name__=='__main__':
	main()