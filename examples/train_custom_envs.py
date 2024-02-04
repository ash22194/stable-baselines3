import torch
from torch import nn
import os
import argparse
from shutil import copyfile
from ruamel.yaml import YAML
import cProfile
from torch.profiler import profile, record_function, ProfilerActivity, schedule

import gymnasium as gym
import numpy as np
from typing import Callable, Dict

from stable_baselines3.gpu_systems import GPUQuadcopter, GPUQuadcopterTT, GPUUnicycle

from stable_baselines3 import A2CwReg, PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback, CustomSaveLogCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, GPUVecEnv
from stable_baselines3.common.learning_schedules import linear_schedule, decay_sawtooth_schedule, exponential_schedule
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

class PytorchProfilerStepCallback(BaseCallback):
	"""
	A callback to periodically profile the training run.
	"""
	def __init__(self, prof, verbose=0):
		super(PytorchProfilerStepCallback, self).__init__(verbose)
		self.prof = prof

	def _on_training_start(self) -> None:
		pass

	def _on_rollout_start(self) -> None:
		pass

	def _on_step(self) -> bool:
		self.prof.step()
		return True

	def _on_rollout_end(self) -> None:
		pass

	def _on_training_end(self) -> None:
		pass


def initialize_model(config_file_path: str, algorithm: str, save_dir: str, run: int = None, env_device: str = 'cpu'):
	
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

	# check batch size
	batch_size = algorithm_args['algorithm_kwargs'].get('batch_size', None)
	if (batch_size is None):
		batch_size = 1
	if (batch_size <= 1):
		batch_size *= (algorithm_args['algorithm_kwargs']['n_steps']*environment_args['num_envs'])
	algorithm_args['algorithm_kwargs']['batch_size'] = int(batch_size)

	# initialize the environment
	num_envs = environment_args.get('num_envs')
	if (env_device=='cuda'):
		if (environment_args.get('name')=='GPUQuadcopter'):
			env = GPUQuadcopter(device=env_device, **(environment_args.get('environment_kwargs', dict())))
		elif (environment_args.get('name')=='GPUQuadcopterTT'):
			env = GPUQuadcopterTT(device=env_device, **(environment_args.get('environment_kwargs', dict())))
		elif (environment_args.get('name')=='GPUUnicycle'):
			env = GPUUnicycle(device=env_device, **(environment_args.get('environment_kwargs', dict())))
		else:
			NotImplementedError
		env = GPUVecEnv(env, num_envs=num_envs)
	else:
		normalized_rewards = environment_args.get('normalized_rewards')
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
		elif (learning_rate_schedule['type'] == 'kla'):
			algorithm_args['algorithm_kwargs']['target_kl'] = learning_rate_schedule['target_kl']
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
			model = A2CwReg(policy_args.get('type', 'MlpPolicy'), env, **algorithm_args.get('algorithm_kwargs'), policy_kwargs=policy_args.get('policy_kwargs'))
		elif (algorithm == 'PPO'):
			model = PPO(policy_args.get('type', 'MlpPolicy'), env, **algorithm_args.get('algorithm_kwargs'), policy_kwargs=policy_args.get('policy_kwargs'))
		elif (algorithm == 'TD3'):
			model = TD3(policy_args.get('type', 'MlpPolicy'), env, **algorithm_args.get('algorithm_kwargs'), policy_kwargs=policy_args.get('policy_kwargs'))

		log_path = os.path.join(save_dir, algorithm, 'tb_log')
		logger = configure_logger(verbose=1, tensorboard_log=log_path, tb_log_name=algorithm, reset_num_timesteps=True)
		# copy config file 
		copyfile(config_file_path, os.path.join(logger.get_dir(), 'cfg.yaml'))
		env_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(config_file_path))))
		if (env_device=='cuda'):
			class_file_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../stable_baselines3/gpu_systems', env_name + '.py')
		elif (env_device=='cpu'):
			class_file_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../stable_baselines3/systems', env_name + '.py')
		else:
			ValueError
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
	parser.add_argument('--env_device', type=str, default='cpu', help='cpu/cuda does the environment run on GPU?')
	parser.add_argument('--profile', default=False, action='store_true', help='profile this run?')

	args = parser.parse_args()
	env_name = args.env_name
	algo = args.algorithm.lower()
	run_id = args.run
	env_device = args.env_device
	profile_run = args.profile

	cfg_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', env_name, env_device, algo, 'sweep_optimized_cfg.yaml')
	if (not os.path.isfile(cfg_abs_path)):
		cfg_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', env_name, env_device, algo, 'cfg.yaml')
	save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', env_name, env_device)

	model, callback, cfg = initialize_model(cfg_abs_path, algo, save_dir, run_id, env_device)

	# train the model
	total_timesteps = cfg['algorithm']['total_timesteps']
	model_load = model.num_timesteps > 0
	log_interval = 1 # number of steps after which logging happens = (num_envs*n_steps_to_update)*log_interval
	if ((total_timesteps-model.num_timesteps) > 0):
		if (profile_run):
			if (env_device=='cuda') or (cfg['algorithm']['algorithm_kwargs'].get('device', 'cpu')=='cuda'):
				activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
				profile_memory = True
				use_cuda = True

				n_steps = cfg['algorithm']['algorithm_kwargs'].get('n_steps', 2000)
				with profile(
					activities=activities, 
					profile_memory=profile_memory, 
					use_cuda=use_cuda, 
					with_stack=True,
					record_shapes=False,
					schedule=schedule(wait=n_steps, warmup=n_steps, active=n_steps, repeat=2),
					on_trace_ready=torch.profiler.tensorboard_trace_handler(model.logger.dir)
				) as prof:
					callback = [callback, PytorchProfilerStepCallback(prof)]
					with record_function('model_learn'):
						model.learn(
							total_timesteps=total_timesteps-model.num_timesteps,
							log_interval=log_interval,
							callback=callback,
							reset_num_timesteps=(not model_load)
						)
				prof.export_chrome_trace(os.path.join(model.logger.dir, 'run_profile.json'))
			
			else:
				prof = cProfile.Profile()
				prof.enable()
				model.learn(
					total_timesteps=total_timesteps-model.num_timesteps,
					log_interval=log_interval,
					callback=callback,
					reset_num_timesteps=(not model_load)
				)
				prof.disable()
				prof.dump_stats(os.path.join(model.logger.dir, 'run_profile'))

		else:
			model.learn(
				total_timesteps=total_timesteps-model.num_timesteps,
				log_interval=log_interval,
				callback=callback,
				reset_num_timesteps=(not model_load)
			)

	# evaluate the model
	eval_envname = cfg['environment'].get('eval_envname', cfg['environment']['name'])
	test_env = gym.make(eval_envname, **cfg['environment'].get('environment_kwargs', dict()))
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