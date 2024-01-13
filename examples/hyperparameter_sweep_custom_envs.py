import torch
from torch import nn
import os
import argparse
from shutil import copyfile
from ruamel.yaml import YAML
from copy import deepcopy
from deepdiff import DeepDiff

import gymnasium as gym
import numpy as np
from typing import Callable, Dict

from stable_baselines3 import A2CwReg, PPO, TD3
from stable_baselines3.gpu_systems import GPUQuadcopter, GPUUnicycle
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, GPUVecEnv
from stable_baselines3.common.learning_schedules import linear_schedule, decay_sawtooth_schedule, exponential_schedule
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb

def run_trial(trial, default_cfg: Dict, sweep_cfg: Dict, save_dir: str, env_device: str):
	# sample a config
	criteria = sweep_cfg.pop('criteria', dict())
	cfg = deepcopy(default_cfg)
	for cfg_type in sweep_cfg.keys():
		assert (cfg_type=='environment') or (cfg_type=='algorithm') or (cfg_type=='policy'), 'hyper-parameter group %s must be of type environment/algorithm/policy' % cfg_type

		for k, v in sweep_cfg[cfg_type].items():
			if (sweep_cfg[cfg_type][k]['type'] == 'int'):
				log = sweep_cfg[cfg_type][k].get('log', False)
				steps = sweep_cfg[cfg_type][k].get('steps', 1)
				sampled_param = {
					k: trial.suggest_int(
						name=k,
						low=sweep_cfg[cfg_type][k]['low'],
						high=sweep_cfg[cfg_type][k]['high'],
						log=log,
						step=steps
					)
				}
			elif (sweep_cfg[cfg_type][k]['type'] == 'float'):
				log = sweep_cfg[cfg_type][k].get('log', False)
				steps = sweep_cfg[cfg_type][k].get('steps', None) 
				sampled_param = {
					k: trial.suggest_float(
						name=k,
						low=sweep_cfg[cfg_type][k]['low'],
						high=sweep_cfg[cfg_type][k]['high'],
						log=log,
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

	# check batch size
	batch_size = cfg['algorithm']['algorithm_kwargs'].get('batch_size', None)
	if (batch_size is None):
		batch_size = 1
	if (batch_size <= 1):
		batch_size *= (cfg['algorithm']['algorithm_kwargs']['n_steps']*cfg['environment']['num_envs'])
	cfg['algorithm']['algorithm_kwargs']['batch_size'] = batch_size

	# save the sampled cfg
	trial_save_dir = os.path.join(save_dir, 'trial_' + str(trial.number))
	assert (not os.path.isdir(trial_save_dir)), 'trial directory already exists!'
	os.mkdir(trial_save_dir)
	with open(os.path.join(trial_save_dir, 'sampled_cfg.yaml'), 'w') as f:
		YAML(typ='rt', pure=True).dump(cfg, f)

	# initialize a model
	model = initialize_model(cfg, trial_save_dir, env_device)

	# define pruning callback
	eval_envname = cfg['environment'].get('eval_envname', cfg['environment']['name'])
	eval_env = gym.make(eval_envname, **cfg['environment'].get('environment_kwargs', dict()))
	eval_every_timestep = cfg['algorithm'].get('save_every_timestep', None)
	if (eval_every_timestep is None):
		eval_every_timestep = cfg['algorithm']['total_timesteps']
	callback = PruneTrialCallback(
		trial=trial, 
		eval_every_timestep=eval_every_timestep, 
		eval_env=eval_env,
		eval_criteria=criteria,
		save_path=trial_save_dir,
		save_prefix=cfg['policy'].get('save_prefix', 'model')
	)

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
	test_env = gym.make(eval_envname, **cfg['environment'].get('environment_kwargs', dict()))
	num_episodes = criteria.get('num_episodes', 20)
	ep_reward, ep_discounted_reward, final_err = evaluate_model(model, test_env, num_episodes)
	eval_metric = criteria.get('type', 'ep_discounted_reward')

	print('Reporting ', eval_metric)
	if (eval_metric == 'ep_reward'):
		return np.mean(ep_reward)
	elif (eval_metric == 'ep_discounted_reward'):
		return np.mean(ep_discounted_reward)
	elif (eval_metric == 'final_err'):
		return -np.mean(np.abs(final_err))

def initialize_model(cfg: Dict, save_dir: str, env_device: str):
	# initialize the policy args
	policy_args = deepcopy(cfg['policy'])
	if ('policy_kwargs' not in policy_args.keys()):
		policy_args['policy_kwargs'] = dict()

	environment_args = deepcopy(cfg['environment'])

	algorithm_args = deepcopy(cfg['algorithm'])
	if ('algorithm_kwargs' not in algorithm_args.keys()):
		algorithm_args['algorithm_kwargs'] = dict()
	if (env_device=='cuda'):
		algorithm_args['algorithm_kwargs']['device'] = 'cuda'

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
	num_envs = environment_args.get('num_envs')
	normalized_rewards = environment_args.get('normalized_rewards', False)
	if (env_device=='cuda'):
		if (environment_args.get('name')=='GPUQuadcopter'):
			env = GPUQuadcopter(device=env_device, **(environment_args.get('environment_kwargs', dict())))
		elif (environment_args.get('name')=='GPUUnicycle'):
			env = GPUUnicycle(device=env_device, **(environment_args.get('environment_kwargs', dict())))
		else:
			NotImplementedError
		env = GPUVecEnv(env, num_envs=num_envs)
	else:
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
	elif (algorithm == 'TD3'):
		model = TD3('MlpPolicy', env, **algorithm_args.get('algorithm_kwargs'), policy_kwargs=policy_args.get('policy_kwargs'))
	logger = configure(folder=save_dir, format_strings=["tensorboard"])
	model.set_logger(logger)

	return model

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

		final_err[ee] = test_env.get_goal_dist()

	return ep_reward, ep_discounted_reward, final_err

class PruneTrialCallback(BaseCallback):
	"""
	A callback to periodically evaluate the model that derives from ``BaseCallback``.

	:param verbose: (int) Verbosity level 0: not output 1: info 2: debug
	"""
	def __init__(self, trial, eval_every_timestep, eval_env, eval_criteria: Dict, save_path: str, save_prefix='model', verbose=0):
		super(PruneTrialCallback, self).__init__(verbose)

		self.trial = trial
		self.eval_every_timestep = eval_every_timestep
		self.eval_env = eval_env
		self.num_eval_episodes = eval_criteria.get('num_episodes', 20)
		self.eval_metric = eval_criteria.get('type', 'ep_discounted_reward')
		self.save_prefix = save_prefix
		self.save_path = save_path

		self.curr_check_point_id = 0

	def _on_training_start(self) -> None:
		pass

	def _on_rollout_start(self) -> None:
		pass

	def _on_step(self) -> bool:
		# log terminal episodic info
		terminal_infos = [info_i for ii, info_i in enumerate(self.locals['infos']) if self.locals['dones'][ii]]
		if (len(terminal_infos) > 0):
			terminal_statnames = [ts for ts in terminal_infos[0].keys() if (ts.startswith('ep_'))]
			for ts in terminal_statnames:
				ts_stat = [terminal_infos[ii][ts] for ii in range(len(terminal_infos)) if (not np.isnan(terminal_infos[ii][ts]))]
				self.logger.record("rollout/" + ts, np.mean(ts_stat))

		# check if trial should be pruned
		check_point_id = divmod(self.model.num_timesteps+1, self.eval_every_timestep)[0]
		if (check_point_id > self.curr_check_point_id):
			ep_reward, ep_discounted_reward, final_err = evaluate_model(self.model, self.eval_env, self.num_eval_episodes)
			self.curr_check_point_id = check_point_id

			print('Reporting ', self.eval_metric)
			if (self.eval_metric == 'ep_reward'):
				self.trial.report(np.mean(ep_reward), self.model.num_timesteps+1)
			elif (self.eval_metric == 'ep_discounted_reward'):
				self.trial.report(np.mean(ep_discounted_reward), self.model.num_timesteps+1)
			elif (self.eval_metric == 'final_err'):
				self.trial.report(-np.mean(np.abs(final_err)), self.model.num_timesteps+1)

			if (self.trial.should_prune()):
				raise optuna.TrialPruned()
			else:
				# if not pruned, save the checkpoint
				save_id = check_point_id*self.eval_every_timestep
				self.model.save(os.path.join(self.save_path, self.save_prefix + '_' + str(save_id)))
			
		return True

	def _on_rollout_end(self) -> None:
		pass

	def _on_training_end(self) -> None:
		pass

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='linearsystem', help='environment name')
	parser.add_argument('--env_device', type=str, default='cpu', help='cpu/cuda simulate environments on cpu or cuda device')
	parser.add_argument('--algorithm', type=str, default='ppo', help='algorithm to train with')
	parser.add_argument('--num_trials', type=int, default=10, help='number of trials to evaluate')

	args = parser.parse_args()
	env_name = args.env_name
	env_device = args.env_device
	algo = args.algorithm.lower()
	num_trials = args.num_trials

	# load cfg files
	default_cfg_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', env_name, env_device, algo, 'cfg.yaml')
	sweep_cfg_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', env_name, env_device, algo, 'sweep_cfg.yaml') 
	if (env_device=='cpu'):
		class_file_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../stable_baselines3/systems', env_name + '.py')
	elif (env_device=='cuda'):
		class_file_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../stable_baselines3/gpu_systems', env_name + '.py')
	else:
		ValueError
	default_cfg = YAML().load(open(default_cfg_abs_path, 'r'))
	if (default_cfg['algorithm'].get('name') is not None):
		assert algo.upper() == default_cfg['algorithm']['name'], 'default config file is for a different algorithm'
	sweep_cfg = YAML().load(open(sweep_cfg_abs_path, 'r'))
	
	# create save directory
	save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', env_name, env_device, algo.upper(), 'sweeps')
	if (not os.path.isdir(save_dir)):
		os.mkdir(save_dir)
	# find the current sweep number
	new_sweep = True
	sweep_count = 0
	for ff in os.listdir(save_dir):
		if ('sweep_' in ff):
			sweep_count += 1
			tt = int(ff.split('sweep_')[-1])
			
			candidate_default_cfg = os.path.join(save_dir, 'sweep_' + str(tt), 'cfg.yaml')
			candidate_sweep_cfg = os.path.join(save_dir, 'sweep_' + str(tt), 'sweep_cfg.yaml')
			if (os.path.isfile(candidate_default_cfg) and os.path.isfile(candidate_sweep_cfg)):
				candidate_default_cfg = YAML().load(open(candidate_default_cfg, 'r'))
				candidate_sweep_cfg = YAML().load(open(candidate_sweep_cfg, 'r'))

				is_not_new_default = (len(DeepDiff(candidate_default_cfg, default_cfg, ignore_order=True))==0)
				is_not_new_sweep = (len(DeepDiff(candidate_sweep_cfg, sweep_cfg, ignore_order=True))==0)
				if ((is_not_new_default) and (is_not_new_sweep)):
					# append to previous sweep
					sweep_number = tt
					new_sweep = False
					break

	if (new_sweep):
		sweep_number = sweep_count + 1
	sweep_dir = os.path.join(save_dir, 'sweep_' + str(sweep_number))
	if (not os.path.isdir(sweep_dir)):
		print('Creating sweep directory')
		os.mkdir(sweep_dir)
	print('Sweep directory :', sweep_dir)

	# copy sweep cfg files
	if (new_sweep):
		copyfile(default_cfg_abs_path, os.path.join(sweep_dir, 'cfg.yaml'))
		copyfile(class_file_abs_path, os.path.join(sweep_dir, env_name + '.py'))
		copyfile(sweep_cfg_abs_path, os.path.join(sweep_dir, 'sweep_cfg.yaml'))

	# create storage
	sweep_storage = JournalStorage(JournalFileStorage(os.path.join(sweep_dir, "sweep_journal.log")))

	# run trials
	study_name = env_name + "_" + env_device + "_" + algo + "_sweep_" + str(sweep_number)
	wandb.login()
	wandb_kwargs = {"project": study_name, "name": study_name}
	wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

	@wandbc.track_in_wandb()
	def _run_trial(tr):
		value = run_trial(tr, default_cfg=deepcopy(default_cfg), sweep_cfg=deepcopy(sweep_cfg), save_dir=sweep_dir, env_device=env_device)
		wandb.log({"value": value})
		return value

	study = optuna.create_study(
		study_name=study_name,
		direction="maximize",
		sampler=TPESampler(),
		pruner=MedianPruner(n_startup_trials=num_trials, n_warmup_steps=1),
		storage=sweep_storage,
		load_if_exists=True
	)
	study.optimize(_run_trial, n_trials=num_trials, n_jobs=1, callbacks=[wandbc])

if __name__=='__main__':
	main()