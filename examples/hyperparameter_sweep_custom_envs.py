import torch
from torch import nn
import os
import argparse
from shutil import copyfile
from ruamel.yaml import YAML
from copy import deepcopy

import gymnasium as gym
import numpy as np
from typing import Callable, Dict

from stable_baselines3 import A2CwReg, PPO
from stable_baselines3.common.callbacks import SaveEveryCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.learning_schedules import linear_schedule, decay_sawtooth_schedule, exponential_schedule
from stable_baselines3.common.logger import configure
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb

def run_trial(trial, default_cfg: Dict, sweep_cfg: Dict, save_dir: str):
	# sample a config
	cfg = deepcopy(default_cfg)
	for cfg_type in sweep_cfg.keys():
		assert (cfg_type=='environment') or (cfg_type=='algorithm') or (cfg_type=='policy'), 'hyper-parameter group %s must be of type environment/algorithm/policy' % cfg_type

		for k, v in sweep_cfg[cfg_type].items():
			if (sweep_cfg[cfg_type][k]['type'] == 'int'):
				steps = sweep_cfg[cfg_type][k].get('steps', 1)
				sampled_param = {
					k: trial.suggest_int(
						name=k,
						low=sweep_cfg[cfg_type][k]['low'],
						high=sweep_cfg[cfg_type][k]['high'],
						step=steps
					)
				}
			elif (sweep_cfg[cfg_type][k]['type'] == 'float'):
				steps = sweep_cfg[cfg_type][k].get('steps', None) 
				sampled_param = {
					k: trial.suggest_float(
						name=k,
						low=sweep_cfg[cfg_type][k]['low'],
						high=sweep_cfg[cfg_type][k]['high'],
						step=steps
					)
				}
			elif (sweep_cfg[cfg_type][k]['type'] == 'categorical'):
				sampled_param = {
					k: trial.suggest_categorical(
						name=k,
						choices=sweep_cfg[cfg_type][k]['choices']
					)
				}

			if (k in cfg[cfg_type].keys()):
				cfg[cfg_type].update(sampled_param)
			elif (k in cfg[cfg_type][cfg_type+'_kwargs'].keys()):
				cfg[cfg_type][cfg_type+'_kwargs'].update(sampled_param)
			else:
				raise KeyError('invalid hyper-parameter %s defined in sweep' % k)

	# make sure the batch_size is an integer fraction of num_envs * n_steps
	cfg['algorithm']['algorithm_kwargs']['batch_size'] = cfg['environment']['num_envs'] * cfg['algorithm']['algorithm_kwargs']['n_steps']

	# save the sampled cfg
	trial_save_dir = os.path.join(save_dir, 'trial_' + str(trial.number))
	assert (not os.path.isdir(trial_save_dir)), 'trial directory already exists!'
	os.mkdir(trial_save_dir)
	with open(os.path.join(trial_save_dir, 'sampled_cfg.yaml'), 'w') as f:
		YAML(typ='rt', pure=True).dump(cfg, f)

	# initialize a model
	model, callback = initialize_model(cfg, trial_save_dir)

	# train the model
	total_timesteps = cfg['algorithm']['total_timesteps']
	model_load = model.num_timesteps > 0
	log_interval = 1 # number of steps after which logging happens = (num_envs*n_steps_to_update)*log_interval
	if ((total_timesteps-model.num_timesteps) > 0):
		model.learn(
			total_timesteps=total_timesteps-model.num_timesteps,
			log_interval=log_interval,
			callback=callback,
			reset_num_timesteps=(not model_load)
		)

	# evaluate the model
	test_env = gym.make(cfg['environment']['name'], **cfg['environment'].get('environment_kwargs', dict()))
	ep_reward, ep_discounted_reward, final_err = evaluate_model(model, test_env, 5)
	
	return np.mean(ep_discounted_reward)

def initialize_model(cfg: Dict, save_dir: str):
	# initialize the policy args
	policy_args = cfg['policy']
	if ('policy_kwargs' not in policy_args.keys()):
		policy_args['policy_kwargs'] = dict()

	if ('activation_fn' in policy_args['policy_kwargs'].keys()):
		if (policy_args['policy_kwargs']['activation_fn'] == 'relu'):
			policy_args['policy_kwargs']['activation_fn'] = nn.ReLU
		elif (policy_args['policy_kwargs']['activation_fn'] == 'elu'):
			# TODO: Add support for ELU 
			pass
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
	environment_args = cfg['environment']
	num_envs = environment_args.get('num_envs')
	env = make_vec_env(environment_args.get('name'), num_envs, env_kwargs=environment_args.get('environment_kwargs', dict()))

	# initialize the agent
	algorithm_args = cfg['algorithm']
	if ('algorithm_kwargs' not in algorithm_args.keys()):
		algorithm_args['algorithm_kwargs'] = dict()
	
	learning_rate_schedule = algorithm_args['algorithm_kwargs'].pop('learning_rate_schedule', None)
	if (learning_rate_schedule is not None):
		# assert type(learning_rate_schedule) is Dict, 'learning rate schedule must be a dict, found it to be of type %s' % type(learning_rate_schedule).__name__
		learning_rate = algorithm_args['algorithm_kwargs'].pop('learning_rate', None)
		assert learning_rate is not None, 'learning rate schedule specified without learning rate'
		
		if (learning_rate_schedule['type'] == 'lin'):
			learning_rate = linear_schedule(learning_rate)
		elif (learning_rate_schedule['type'] == 'exp'):
			learning_rate = exponential_schedule(learning_rate, learning_rate_schedule['decay_rate'])
		elif (learning_rate_schedule['type'] == 'sawt'):
			learning_rate = decay_sawtooth_schedule(learning_rate, learning_rate_schedule['sawtooth_width'])
		algorithm_args['algorithm_kwargs']['learning_rate'] = learning_rate

	algorithm = algorithm_args.get('name')
	if (algorithm == 'A2C'):
		model = A2CwReg('MlpPolicy', env, **algorithm_args.get('algorithm_kwargs'), policy_kwargs=policy_args.get('policy_kwargs'))
	elif (algorithm == 'PPO'):
		model = PPO('MlpPolicy', env, **algorithm_args.get('algorithm_kwargs'), policy_kwargs=policy_args.get('policy_kwargs'))
	
	logger = configure(folder=save_dir, format_strings=["tensorboard"])
	model.set_logger(logger)

	save_every_timestep = algorithm_args.get('save_every_timestep', None)
	if (save_every_timestep is None):
		save_every_timestep = algorithm_args.get('total_timesteps')
	callback = SaveEveryCallback(save_every_timestep=save_every_timestep, save_path=logger.get_dir(), save_prefix=policy_args.get('save_prefix'))

	return model, callback

def evaluate_model(model, test_env: gym.Env, num_episodes: int):
	ep_reward = np.zeros(num_episodes)
	ep_discounted_reward = np.zeros(num_episodes)
	final_err = np.zeros(num_episodes)

	for ee in range(num_episodes):
		obs, _ = test_env.reset()
		done = False
		discount = 1
		while (not done):
			action, _state = model.predict(obs, deterministic=True)
			obs, reward, done, _, info = test_env.step(action)

			ep_reward[ee] += reward
			ep_discounted_reward[ee] += (discount*reward)
			discount *= model.gamma

		final_err[ee] = np.linalg.norm(obs.reshape(-1) - test_env.goal.reshape(-1))

	return ep_reward, ep_discounted_reward, final_err

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='linearsystem', help='environment name')
	parser.add_argument('--num_trials', type=int, default=10, help='number of trials to evaluate')
	parser.add_argument('--num_jobs', type=int, default=1, help='number of parallel jobs to run (> 1 requires a DB to be running)')

	args = parser.parse_args()
	env_name = args.env_name
	num_trials = args.num_trials
	num_jobs = args.num_jobs

	# create save directory
	save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', env_name, 'sweeps')
	if (not os.path.isdir(save_dir)):
		os.mkdir(save_dir)
	# find the current sweep number
	sweep_number = 0
	for ff in os.listdir(save_dir):
		if ('sweep_' in ff):
			tt = int(ff.split('sweep_')[-1])
			if (tt > sweep_number):
				sweep_number = tt
	sweep_number += 1
	sweep_dir = os.path.join(save_dir, 'sweep_' + str(sweep_number))
	os.mkdir(sweep_dir)

	# copy sweep cfg files 
	default_cfg_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', env_name, 'cfg.yaml')
	sweep_cfg_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', env_name, 'sweep_cfg.yaml') 
	copyfile(default_cfg_abs_path, os.path.join(sweep_dir, 'cfg.yaml'))
	copyfile(sweep_cfg_abs_path, os.path.join(sweep_dir, 'sweep_cfg.yaml'))

	# load cfg files
	default_cfg = YAML().load(open(default_cfg_abs_path, 'r'))
	sweep_cfg = YAML().load(open(sweep_cfg_abs_path, 'r'))

	# run trials
	_run_trial = lambda tr: run_trial(tr, default_cfg=default_cfg, sweep_cfg=sweep_cfg, save_dir=sweep_dir)
	# instantiate a sampler
	study = optuna.create_study(direction="maximize")

	wandb.login()
	wandb_kwargs = {"project": env_name + '_sweep_' + str(sweep_number)}
	wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)
	
	study.optimize(_run_trial, n_trials=num_trials, n_jobs=num_jobs, callbacks=[wandbc])

if __name__=='__main__':
	main()