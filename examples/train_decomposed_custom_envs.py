import torch
from torch import nn
import os
import warnings
import argparse
from copy import deepcopy
from shutil import copyfile
from ruamel.yaml import YAML
import cProfile
from torch.profiler import profile, record_function, ProfilerActivity, schedule

import gymnasium as gym
import numpy as np
from typing import Callable, Dict

from stable_baselines3.gpu_systems import GPUQuadcopter, GPUQuadcopterTT, GPUQuadcopterDecomposition, GPUUnicycle

from stable_baselines3 import A2CwReg, PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback, CustomSaveLogCallback, CustomEvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, GPUVecEnv
from stable_baselines3.common.learning_schedules import linear_schedule, decay_sawtooth_schedule, exponential_schedule
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_latest_run_id
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

def initialize_environment(environment_args, env_device: str= 'cpu'):

	num_envs = environment_args.get('num_envs')
	if ((env_device=='cuda') or (env_device=='mps')):
		if (environment_args.get('name')=='GPUQuadcopter'):
			env = GPUQuadcopter(device=env_device, **(environment_args.get('environment_kwargs', dict())))
		elif (environment_args.get('name')=='GPUQuadcopterTT'):
			env = GPUQuadcopterTT(device=env_device, **(environment_args.get('environment_kwargs', dict())))
		elif (environment_args.get('name')=='GPUQuadcopterDecomposition'):
			env = GPUQuadcopterDecomposition(device=env_device, **(environment_args.get('environment_kwargs', dict())))
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
				norm_obs=False, # Batch normalization
				norm_reward=True,
				training=True,
				gamma=algorithm_args['algorithm_kwargs'].get('gamma', 0.99)
			)

	return env

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

def parse_decomposition_args(decomposition_args):

	input_adj_mapping = np.array(decomposition_args.pop('in_tree'), dtype=np.int32)
	input_state_mapping = np.array(decomposition_args.pop('in_st_map'), dtype=np.int32)

	n_in = input_state_mapping.shape[0]
	n_st = input_state_mapping.shape[1]
	independent_states = decomposition_args.get('independent_states', None)
	if (independent_states is None):
		independent_states = np.ones(n_st)
	else:
		independent_states = np.array(independent_states)

	assert input_adj_mapping.shape[0]==n_in, 'Check number of inputs in input_tree'
	queue = [[np.array([-1]), None]]
	input_tree = []
	while len(queue) > 0:
		curr_node = queue.pop()
		curr_parent = curr_node[1]
		curr_node = curr_node[0]
		if ((curr_node.shape[0] == 1) and (curr_node[0] == -1)):
			curr_state = np.zeros(n_st)
		else:
			curr_state = input_state_mapping[curr_node[0],:]
			assert np.all((independent_states - curr_state) >=0), 'subsystem includes state variable not part of the independent ones'

		curr_children = np.nonzero(np.any(input_adj_mapping[:,0:1] == curr_node, axis=1))[0]
		coupled_children = []
		while curr_children.shape[0] > 0:
			curr_coupled_children = np.nonzero(np.all(input_adj_mapping[curr_children[0], :] == input_adj_mapping, axis=1))[0]
			curr_children = curr_children[np.logical_not(np.any(curr_coupled_children == curr_children[:,np.newaxis], axis=1))]
			coupled_children += [curr_coupled_children]

		input_tree += [[curr_node, dict(state=curr_state, children=coupled_children, parent=curr_parent)]]
		if (len(coupled_children) > 0):
			queue += [[np.sort(cc), curr_node] for cc in coupled_children]

	decomposition_args['input_tree'] = input_tree
	decomposition_args['independent_states'] = independent_states
	decomposition_args['num_inputs'] = n_in
	decomposition_args['num_states'] = n_st

	return decomposition_args

def get_gpu_env(env_name: str, env_device: str, env_kwargs: Dict = dict()):

	if (env_name=='GPUQuadcopter'):
		env = GPUQuadcopter(device=env_device, **env_kwargs)
	elif (env_name=='GPUQuadcopterTT'):
		env = GPUQuadcopterTT(device=env_device, **env_kwargs)
	elif (env_name=='GPUQuadcopterDecomposition'):
		env = GPUQuadcopterDecomposition(device=env_device, **env_kwargs)
	elif (env_name=='GPUUnicycle'):
		env = GPUUnicycle(device=env_device, **env_kwargs)
	else:
		env = None
	return env

def setup_subpolicy_computation(node_environment_args: dict, node_algorithm_args: dict, node_policy_args: dict, subpolicy_save_dir: str, frozen_weights: dict, env_device: str = 'cpu', save_model: bool = False):

	# create environment
	# update environment config
	# TODO: specify which inputs and state varaibles to fix in the dynamics
	num_envs = node_environment_args.get('num_envs')
	# eval_envname = node_environment_args.get('eval_envname', node_environment_args.get('name'))
	eval_envname = node_environment_args.get('name')
	n_eval_episodes = node_algorithm_args.get('n_eval_episodes', 30)
	if ((env_device=='cuda') or (env_device=='mps')):
		env = get_gpu_env(node_environment_args.get('name'), env_device, node_environment_args.get('environment_kwargs', dict()))
		env = GPUVecEnv(env, num_envs=num_envs)

		eval_env = get_gpu_env(eval_envname, env_device, node_environment_args.get('environment_kwargs', dict()))
	else:
		normalized_rewards = node_environment_args.get('normalized_rewards')
		env = make_vec_env(
			node_environment_args.get('name'), n_envs=num_envs, 
			env_kwargs=node_environment_args.get('environment_kwargs', dict()),
			vec_env_cls=DummyVecEnv
		)
		if (normalized_rewards):
			env = VecNormalize(
				env,
				norm_obs=False,
				norm_reward=True,
				training=True,
				gamma=node_algorithm_args['algorithm_kwargs'].get('gamma', 0.99)
			)

		eval_env = gym.make(eval_envname, **node_environment_args['environment_kwargs'])

	if (eval_env is None):
		eval_env = gym.make(eval_envname, **(node_environment_args.get('environment_kwargs', dict())))
	else:
		eval_env.set_num_envs(n_eval_episodes)
		n_eval_episodes = 1

	# check if saved model exist to load and restart training from
	model_save_prefix = node_policy_args.get('save_prefix')
	files = [f for f in os.listdir(subpolicy_save_dir) if (os.path.isfile(os.path.join(subpolicy_save_dir, f)) and (model_save_prefix in f))]
	if (len(files) > 0):
		save_timestep = np.max([int(f.split('_')[-1].split('.')[0]) for f in files]+[0])
	else:
		save_timestep = 0

	algorithm = node_algorithm_args['name']
	if (save_timestep > 0):
		if (algorithm == 'A2C'):
			model = A2CwReg.load(os.path.join(subpolicy_save_dir, model_save_prefix + '_' + str(save_timestep)))
		elif (algorithm == 'PPO'):
			model = PPO.load(os.path.join(subpolicy_save_dir, model_save_prefix + '_' + str(save_timestep)))
		elif (algorithm == 'TD3'):
			model = TD3.load(os.path.join(subpolicy_save_dir, model_save_prefix + '_' + str(save_timestep)))
		model.set_env(env)

	else:
		node_net_arch = node_policy_args['policy_kwargs']['net_arch']
		# create model
		if (algorithm == 'A2C'):
			model = A2CwReg('DecompositionMlpPolicy', env, **node_algorithm_args.get('algorithm_kwargs'), policy_kwargs=node_policy_args.get('policy_kwargs'))
		elif (algorithm == 'PPO'):
			model = PPO('DecompositionMlpPolicy', env, **node_algorithm_args.get('algorithm_kwargs'), policy_kwargs=node_policy_args.get('policy_kwargs'))
		elif (algorithm == 'TD3'):
			model = TD3('DecompositionMlpPolicy', env, **node_algorithm_args.get('algorithm_kwargs'), policy_kwargs=node_policy_args.get('policy_kwargs'))

		# initialize frozen model weights
		node_weights = model.policy.get_weights()
		weights_load_dict = dict()
		for imma, mma in enumerate(node_net_arch['pi']):
			if mma[3] == 'f':
				ifw_pn = np.nonzero([np.all(np.any(np.array(mma[0])[:,np.newaxis]==fw[0], axis=0)) for fw in frozen_weights['policy_net']])[0]
				assert ifw_pn.shape[0] == 1, 'frozen policy_net weights not found or multiple copies found'
				ifw_pn = ifw_pn[0]
				for iw, ww in enumerate(node_weights['policy_net'][imma]):
					weights_load_dict['mlp_extractor.policy_net.%d.'%imma + ww[0]] = frozen_weights['policy_net'][ifw_pn][1][iw][1]

				ifw_an = np.nonzero([np.all(np.any(np.array(mma[0])[:,np.newaxis]==fw[0], axis=0)) for fw in frozen_weights['action_net']])[0]
				assert ifw_an.shape[0] == 1, 'frozen action_net weights not found or multiple copies found'
				ifw_an = ifw_an[0]
				for iw, ww in enumerate(node_weights['action_net'][imma]):
					weights_load_dict['action_net.%d.'%imma + ww[0]] = frozen_weights['action_net'][ifw_an][1][iw][1]
		model.policy.set_weights(weights_load_dict)

		logger = configure(subpolicy_save_dir, format_strings=["stdout", "tensorboard"])
		model.set_logger(logger)

	save_every_timestep = node_algorithm_args.get('save_every_timestep', node_algorithm_args['total_timesteps'])
	# callback = CustomSaveLogCallback(
	# 	save_every_timestep=save_every_timestep, 
	# 	save_path=subpolicy_save_dir, 
	# 	save_prefix=node_policy_args.get('save_prefix'),
	# 	# termination=dict(criteria='reward', threshold=0.025, repeat=10)
	# )

	stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=node_algorithm_args.get('max_no_improvement_evals', 5), min_evals=node_algorithm_args.get('min_evals', 5), verbose=1)
	callback = CustomEvalCallback(eval_env, eval_freq=int(save_every_timestep/num_envs), n_eval_episodes=n_eval_episodes, log_path=logger.get_dir(), callback_after_eval=stop_train_callback, verbose=1, save_model=save_model)

	return model, callback

def compute_decomposition_policies(config_file_path: str, algorithm: str, save_dir: str, run: int = None, env_device: str = 'cpu', save_model: bool = False):
	
	env_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(config_file_path))))
	if ((env_device=='cuda') or (env_device=='mps')):
		class_file_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../stable_baselines3/gpu_systems', env_name + '.py')
	elif (env_device=='cpu'):
		class_file_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../stable_baselines3/systems', env_name + '.py')
	else:
		ValueError

	algorithm = algorithm.upper()
	save_dir = os.path.join(save_dir, algorithm, 'tb_log')
	if (run is not None):
		# load from a previous run
		run_path = os.path.join(save_dir, f"{algorithm}_{run}")
		assert os.path.isdir(run_path), 'run directory does not exist!'
		save_dir = run_path
		if (os.path.isfile(os.path.join(save_dir, 'cfg.yaml'))):
			config_file_path = os.path.join(save_dir, 'cfg.yaml')
	else:
		latest_run_id = get_latest_run_id(log_path=save_dir, log_name=algorithm)
		save_dir = os.path.join(save_dir, f"{algorithm}_{latest_run_id + 1}")
		os.makedirs(save_dir, exist_ok=False)

	# copy config and environment files
	copyfile(config_file_path, os.path.join(save_dir, 'cfg.yaml'))
	copyfile(class_file_abs_path, os.path.join(save_dir, env_name + '.py'))

	# load the config file and extract hyperparameters
	cfg_orig = YAML().load(open(config_file_path, 'r'))
	cfg = parse_common_args(deepcopy(cfg_orig))
	environment_args = cfg['environment']
	algorithm_args = cfg['algorithm']
	if not (algorithm_args['name'].upper()==algorithm):
		warnings.warn('The config file corresponds to %s but requested training algorithm is %s'%(algorithm_args['name'].upper(), algorithm))
		algorithm_args['name'] = algorithm
	policy_args = cfg['policy']
	net_arch = policy_args['policy_kwargs']['net_arch']
	assert ('pi' in net_arch.keys()), 'net_arch must be a dict with entries pi for policy and optionally vf for value function architectures'
	net_arch['pi'] = [[np.sort(mm[0]).tolist(), mm[1]] for mm in net_arch['pi']]

	decomposition = parse_decomposition_args(cfg['decomposition'])
	input_tree = decomposition['input_tree']
	# populate the input-tree with appropriate module architecture
	for imm in range(len(input_tree)-1):
		mm_inputs = input_tree[imm+1][0]
		which_module = [np.any(mm_inputs[:,np.newaxis] == np.array(archm[0]), axis=0) for archm in net_arch['pi']]
		assert np.all([np.logical_or(np.all(wm), np.logical_not(np.any(wm))) for wm in which_module]), 'inconsistent input coupling in network architecture and decomposition'
		input_tree[imm+1][1]['arch'] = [net_arch['pi'][iwm] for iwm, wm in enumerate(which_module) if np.all(wm)]

	num_inputs = decomposition['num_inputs']
	num_states = decomposition['num_states']
	independent_states = decomposition.pop('independent_states')

	# contruct subsystems and then train policies for subsystems
	num_leaf_nodes = 1
	while (num_leaf_nodes > 0):
		num_leaf_nodes = sum([True for it in input_tree[1:] if ((len(it[1]['children'])==0) and (not ('weights' in it[1].keys())))])

		if (num_leaf_nodes > 0):
			for inode, node in enumerate(input_tree):
				if (len(node[1]['children']) == 0) and (not ('weights' in node[1].keys())):
					# is a leaf node
					node_parent = node[1]['parent']
					inode_parent = np.nonzero([np.all(np.any(mm[0][:,np.newaxis]==node_parent, axis=1)) for mm in input_tree])[0]
					assert inode_parent.shape[0]==1, 'node must have a single parent'
					inode_parent = inode_parent[0]
					break
		else:
			inode = 0
			node = input_tree[inode]

		active_inputs = node[0]
		active_inputs = active_inputs[np.logical_not(active_inputs == -1)]
		frozen_inputs = node[1].get('sub_policies', [])
		if (len(frozen_inputs) > 0):
			frozen_inputs = np.concatenate(tuple(frozen_inputs))
		else:
			frozen_inputs = np.array(frozen_inputs)
		constant_inputs = np.arange(num_inputs)
		constant_inputs = constant_inputs[np.logical_not(
			np.logical_or(
				np.any(constant_inputs[:,np.newaxis] == active_inputs, axis=1), 
				np.any(constant_inputs[:,np.newaxis] == frozen_inputs, axis=1)
			)
		)]
		active_states = node[1]['state']

		# modify the environment args
		node_environment_args = deepcopy(environment_args)

		environment_params = node_environment_args['environment_kwargs'].get('param', dict())
		environment_params['U_DIMS_FREE'] = active_inputs
		environment_params['U_DIMS_CONTROLLED'] = frozen_inputs
		environment_params['U_DIMS_FIXED'] = constant_inputs
		environment_params['X_DIMS_FREE'] = np.nonzero(active_states)[0]
		environment_params['X_DIMS_FIXED'] = np.nonzero(np.logical_and(independent_states, np.logical_not(active_states > 0)))[0]
		node_environment_args['param'] = environment_params

		# copy the algorithm args
		node_algorithm_args = deepcopy(algorithm_args)

		# modify the policy args
		node_policy_args = deepcopy(policy_args)
		node_net_arch = []
		node_vf_arch = [[], []]
		if (active_inputs.shape[0]==0):
			node_vf_arch[0] = ['all']

		frozen_weights = dict()
		frozen_weights['policy_net'] = []
		frozen_weights['action_net'] = []
		for imm, mm in enumerate(input_tree[1:]):
			mm_inputs = mm[0]
			mm_states = mm[1]['state']
			mm_archs = mm[1]['arch']
			# active inputs to train policies for 
			is_mm_active = np.any(active_inputs[:,np.newaxis]==mm_inputs, axis=0)
			if np.any(is_mm_active):
				assert np.all(is_mm_active), 'not all actions in the module are active'
				node_vf_arch[0] += np.nonzero(mm_states)[0].tolist()
				for mma in mm_archs:
					node_net_arch += [[mma[0], np.nonzero(mm_states)[0].tolist(), mma[1], 'a']]
					node_vf_arch[1] = [ll + node_vf_arch[1][il] for il, ll in enumerate(mma[1]) if il < len(node_vf_arch[1])]
					node_vf_arch[1] += mma[1][len(node_vf_arch[1]):]

			# frozen inputs for which policies are pre-trained
			is_mm_frozen = np.any(frozen_inputs[:,np.newaxis]==mm_inputs, axis=0)
			if np.any(is_mm_frozen):
				assert np.all(is_mm_frozen), 'not all actions in the module are frozen'
				node_net_arch += [[mma[0], np.nonzero(mm_states)[0].tolist(), mma[1], 'f'] for mma in mm_archs]
				frozen_weights['policy_net'] += mm[1]['weights']['policy_net']
				frozen_weights['action_net'] += mm[1]['weights']['action_net']

			# constant input
			is_mm_constant = np.any(constant_inputs[:,np.newaxis]==mm_inputs, axis=0)
			if np.any(is_mm_constant):
				assert np.all(is_mm_constant), 'not all actions in the module are constant'
				node_net_arch += [[mma[0], np.nonzero(mm_states)[0].tolist(), [], 'c'] for mma in mm_archs]
		if ('vf' in net_arch.keys()):
			node_policy_args['policy_kwargs']['net_arch'] = dict(pi=node_net_arch, vf=net_arch['vf'])
		else:
			node_vf_arch[0] = np.unique(node_vf_arch[0]).tolist()
			node_policy_args['policy_kwargs']['net_arch'] = dict(pi=node_net_arch, vf=node_vf_arch)

		# where to log trained subpolicy computation?
		if (active_inputs.shape[0] > 0):
			is_root = False
			subpolicy_save_dir = os.path.join(save_dir, 'U'+''.join(str(e) for e in active_inputs.tolist())+'_X'+''.join(str(e) for e in np.nonzero(node[1]['state'])[0].tolist()))
			os.makedirs(subpolicy_save_dir, exist_ok=True)
		else:
			is_root = True
			subpolicy_save_dir = save_dir

		# setup model for computing subpolicies
		# dump the node config
		node_cfg = deepcopy(cfg_orig)
		node_cfg.pop('decomposition')
		node_cfg['environment'] = deepcopy(node_environment_args)
		for key_param, param in node_cfg['environment']['environment_kwargs']['param'].items():
			if (type(param)==np.ndarray):
				node_cfg['environment']['environment_kwargs']['param'][key_param] = param.tolist()

		node_cfg['policy']['policy_kwargs']['net_arch'] = node_net_arch
		with open(os.path.join(subpolicy_save_dir, 'node_cfg.yaml'), 'w') as f:
			YAML(typ='rt', pure=True).dump(node_cfg, f)

		model, callback = setup_subpolicy_computation(node_environment_args, node_algorithm_args, node_policy_args, subpolicy_save_dir, frozen_weights, env_device, save_model)

		# If any active inputs then train
		if (not is_root):
			# train
			total_timesteps = node_algorithm_args.get('total_timesteps')
			save_every_timestep = node_algorithm_args.get('save_every_timestep', total_timesteps)
			log_interval = 1

			model.learn(
				total_timesteps=total_timesteps-model.num_timesteps,
				log_interval=log_interval,
				callback=callback,
				reset_num_timesteps=(not (model.num_timesteps > 0))
			)

		else:
			# reached root node!
			# evaluate the model
			eval_envname = node_environment_args.get('eval_envname', node_environment_args['name'])
			test_env = gym.make(eval_envname, **node_environment_args.get('environment_kwargs', dict()))
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

			# save the model
			model.save(os.path.join(subpolicy_save_dir, node_policy_args.get('save_prefix') + '_final'))

			# check if the frozen weights are the same as the saved ones
			# state_dict = model.policy.state_dict()
			# policy_net_err = dict()
			# action_net_err = dict()
			# for imma, mma in enumerate(node_net_arch):
			# 	inode_mma = np.nonzero([np.all(np.any(ino[0][:,np.newaxis] == np.array(mma[0]), axis=0)) for ino in input_tree])[0][0]
			# 	inode_sub_mma = np.nonzero([np.all(np.any(ipn[0][:,np.newaxis]==np.array(mma[0]), axis=0)) for ipn in input_tree[inode_mma][1]['weights_tensor']['policy_net']])[0][0]

			# 	for iw, ww in enumerate(input_tree[inode_mma][1]['weights_tensor']['policy_net'][inode_sub_mma][1]):
			# 		w_id = 'mlp_extractor.policy_net.%d.'%(imma)+ww[0]
			# 		policy_net_err[w_id] = ww[1] - state_dict[w_id]

			# 	for iw, ww in enumerate(input_tree[inode_mma][1]['weights_tensor']['action_net'][inode_sub_mma][1]):
			# 		w_id = 'action_net.%d.'%(imma)+ww[0]
			# 		action_net_err[w_id] = ww[1] - state_dict[w_id]

			return model

		# Update the parent in the input_tree and save subpolicies for reuse
		parent_children = input_tree[inode_parent][1]['children']
		# remove the current node from the parent_children list and add subpolicy weights
		inode_parent_children = np.nonzero([np.all(np.any(pc[:,np.newaxis]==active_inputs, axis=1)) for pc in parent_children])[0]

		assert inode_parent_children.shape[0]==1, 'current node must correspond to a single child in the parent node'

		parent_children.pop(inode_parent_children[0])
		input_tree[inode_parent][1]['children'] = parent_children
		input_tree[inode_parent][1]['state'] = np.logical_or(input_tree[inode_parent][1]['state'], node[1]['state']).astype(np.int32)

		inode_parent_subpolicies = input_tree[inode_parent][1].get('sub_policies', [])
		inode_parent_subpolicies += [active_inputs]
		inode_parent_subpolicies += node[1].get('sub_policies', [])
		input_tree[inode_parent][1]['sub_policies'] = inode_parent_subpolicies

		trained_weights = model.policy.get_weights()

		input_tree[inode][1]['weights'] = dict() # save weights corresponding to the active module
		input_tree[inode][1]['weights']['policy_net'] = []
		input_tree[inode][1]['weights']['action_net'] = []
		# input_tree[inode][1]['weights_tensor'] = dict()
		# input_tree[inode][1]['weights_tensor']['policy_net'] = []
		# input_tree[inode][1]['weights_tensor']['action_net'] = []
		for imma, mma in enumerate(node_net_arch):
			if (np.all(np.any(active_inputs[:,np.newaxis] == np.array(mma[0]), axis=0))):
				input_tree[inode][1]['weights']['policy_net'] += [[np.array(mma[0]), [(ww[0], ww[1].clone()) for ww in trained_weights['policy_net'][imma]]]]
				input_tree[inode][1]['weights']['action_net'] += [[np.array(mma[0]), [(ww[0], ww[1].clone()) for ww in trained_weights['action_net'][imma]]]]
				# input_tree[inode][1]['weights_tensor']['policy_net'] += [[np.array(mma[0]), [(ww[0], ww[1].detach()) for ww in trained_weights['policy_net'][imma]]]]
				# input_tree[inode][1]['weights_tensor']['action_net'] += [[np.array(mma[0]), [(ww[0], ww[1].detach()) for ww in trained_weights['action_net'][imma]]]]
		del model

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='quadcopter_decomposition', help='environment name')
	parser.add_argument('--algorithm', type=str, default='ppo', help='algorithm to train')
	parser.add_argument('--run', type=int, default=None, help='run number')
	parser.add_argument('--env_device', type=str, default='cpu', help='cpu/cuda does the environment run on GPU?')
	parser.add_argument('--save_intermittent_model', default=False, action='store_true', help='save intermittent models?')
	# parser.add_argument('--profile', default=False, action='store_true', help='profile this run?')

	args = parser.parse_args()
	env_name = args.env_name
	algo = args.algorithm.lower()
	run_id = args.run
	env_device = args.env_device
	device_dir = env_device
	if (env_device == 'mps'):
		device_dir = 'cuda'
	elif not ((env_device=='cuda') or (env_device=='cpu')):
		NotImplementedError
	save_model = args.save_intermittent_model
	# profile_run = args.profile

	cfg_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', env_name, device_dir, algo, 'sweep_optimized_cfg.yaml')
	if (not os.path.isfile(cfg_abs_path)):
		cfg_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', env_name, device_dir, algo, 'cfg.yaml')
	save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', env_name, device_dir)

	compute_decomposition_policies(config_file_path=cfg_abs_path, algorithm=algo, save_dir=save_dir, run=run_id, env_device=env_device, save_model=save_model)

if __name__=='__main__':
	main()