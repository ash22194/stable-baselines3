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
from ipdb import set_trace
from matplotlib import pyplot as plt
import gymnasium as gym
import numpy as np
from typing import Callable, Dict

from stable_baselines3.gpu_systems import GPUQuadcopter, GPUQuadcopterTT, GPUUnicycle

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
	if (env_device=='cuda'):
		algorithm_args['algorithm_kwargs']['device'] = 'cuda'
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


def find_next_load_file(load_dir: str, current_load_iter: int, return_max=False):
	# looks through zip files with format *_<iter_number>.zip
	# if return_max is true returns the latest checkpoint
	# if return_max is false returns the earliest checkpoint greater than current_load_iter
	# if no checkpoint exists beyond the current_load_iter, behavior is similar to return_max=true

	files = [ff for ff in os.listdir(load_dir) if (os.path.splitext(ff)[-1]=='.zip')]
	assert len(files)>0, "no files with .zip extension found to load from"
	files = np.array(files)[np.argsort([int(os.path.splitext(ff)[0].split('_')[-1]) for ff in files])].tolist()

	# finds file with iter_number > current_load_iter closest to current_load_iter
	existing_files_above_current = [ff for ff in files if (int(os.path.splitext(ff)[0].split('_')[-1]) > current_load_iter)]
	if ((len(existing_files_above_current)==0) or (return_max)):
		current_load_iter = int(os.path.splitext(files[-1])[0].split('_')[-1])
		return os.path.join(load_dir, files[-1]), current_load_iter
	else:
		current_load_iter = int(os.path.splitext(existing_files_above_current[0])[0].split('_')[-1])
		return os.path.join(load_dir, existing_files_above_current[0]), current_load_iter


def evaluate_decomposition_policies(load_dir: str, env_device: str = 'cpu', test_starts: str = None):

	assert os.path.isdir(load_dir), "load_dir must be a directory"

	# load config file
	files = [os.path.join(load_dir, f) for f in os.listdir(load_dir) if os.path.isfile(os.path.join(load_dir, f))]
	cfg_files = [ff for ff in files if (os.path.splitext(ff)[-1] == '.yaml')]
	if (len(cfg_files) > 1):
		cfg_file = [os.path.splitext(os.path.basename(ff))[0]=='cfg' for ff in cfg_files]
		if (np.any(cfg_file)):
			cfg_file = cfg_files[np.nonzero(cfg_file)[0][0]]
		else:
			assert False, 'No cfg.yaml file found'
	else:
		cfg_file = cfg_files[0]
	cfg = YAML().load(open(cfg_file, 'r'))
	cfg = parse_common_args(cfg)

	# extract decomposition args
	policy_args = cfg['policy']
	net_arch = policy_args['policy_kwargs']['net_arch']

	decomposition = parse_decomposition_args(cfg['decomposition'])
	num_inputs = decomposition['num_inputs']
	num_states = decomposition['num_states']
	input_tree = decomposition['input_tree']
	# populate the input-tree with appropriate module architecture
	for imm in range(len(input_tree)-1):
		mm_inputs = input_tree[imm+1][0]
		which_module = [np.any(mm_inputs[:,np.newaxis] == np.array(archm[0]), axis=0) for archm in net_arch['pi']]
		assert np.all([np.logical_or(np.all(wm), np.logical_not(np.any(wm))) for wm in which_module]), 'inconsistent input coupling in network architecture and decomposition'
		input_tree[imm+1][1]['arch'] = [net_arch['pi'][iwm] for iwm, wm in enumerate(which_module) if np.all(wm)]
	independent_states = decomposition.pop('independent_states')

	# extract environment args
	environment_args = cfg['environment']
	environment_params = environment_args['environment_kwargs'].get('param', dict())
	environment_params['U_DIMS_FREE'] = np.array([], dtype=np.int32)
	environment_params['U_DIMS_CONTROLLED'] = np.arange(num_inputs, dtype=np.int32)
	environment_params['U_DIMS_FIXED'] = np.array([], dtype=np.int32)
	environment_params['X_DIMS_FREE'] = np.nonzero(independent_states)[0]
	environment_params['X_DIMS_FIXED'] = np.array([], dtype=np.int32)
	environment_args['environment_kwargs']['param'] = environment_params
	if (env_device=='cpu'):
		env = gym.make(environment_args['name'], **environment_args['environment_kwargs'])
	else:
		NotImplementedError

	# load starts if test_starts exist else sample and save
	if (os.path.isfile(test_starts) and (os.path.splitext(test_starts)[-1]=='.npy')):
		starts = np.load(test_starts, allow_pickle=False, mmap_mode=None)
	else:
		num_starts = 100
		starts = []
		for nn in range(num_starts):
			env.reset()
			starts += [env.state.copy()]
		starts = np.array(starts)
		np.save(os.path.join(load_dir, 'test_starts.npy'), starts, allow_pickle=False)

	algorithm_args = cfg['algorithm']
	if (algorithm_args['name'] == 'PPO'):
		algorithm = PPO
	else:
		NotImplementedError

	# load subsystem policies at different checkpoints to run evaluation of full policy
	mean_rewards = []
	mean_discounted_rewards = []
	mean_final_errors = []
	checkpoint_steps = []
	previous_load_iter = -1
	current_load_iter = 0
	cummulative_steps = 0
	while (not (current_load_iter == previous_load_iter)):
		# load model weights for the current iteration
		curr_input_tree = deepcopy(input_tree)
		num_leaf_nodes = 1
		while (num_leaf_nodes > 0):
			num_leaf_nodes = sum([True for it in curr_input_tree[1:] if ((len(it[1]['children'])==0) and (not ('weights' in it[1].keys())))])

			if (num_leaf_nodes > 0):
				for inode, node in enumerate(curr_input_tree):
					if (len(node[1]['children']) == 0) and (not ('weights' in node[1].keys())):
						# is a leaf node
						node_parent = node[1]['parent']
						inode_parent = np.nonzero([np.all(np.any(mm[0][:,np.newaxis]==node_parent, axis=1)) for mm in curr_input_tree])[0]
						assert inode_parent.shape[0]==1, 'node must have a single parent'
						inode_parent = inode_parent[0]
						break
			else:
				node = curr_input_tree[0]

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

			# subpolicy architecture
			node_net_arch = []
			frozen_weights = dict()
			frozen_weights['policy_net'] = []
			frozen_weights['action_net'] = []
			for imm, mm in enumerate(curr_input_tree[1:]):
				mm_inputs = mm[0]
				mm_states = mm[1]['state']
				mm_archs = mm[1]['arch']
				# active inputs to train policies for 
				is_mm_active = np.any(active_inputs[:,np.newaxis]==mm_inputs, axis=0)
				if np.any(is_mm_active):
					assert np.all(is_mm_active), 'not all actions in the module are active'
					node_net_arch += [[mma[0], np.nonzero(mm_states)[0].tolist(), mma[1], 'a'] for mma in mm_archs]
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

			if (active_inputs.shape[0] > 0):
				subpolicy_load_dir = os.path.join(load_dir, 'U'+''.join(str(e) for e in active_inputs.tolist())+'_X'+''.join(str(e) for e in np.nonzero(node[1]['state'])[0].tolist()), 'evaluations')
				assert os.path.isdir(subpolicy_load_dir), 'looking for subpolicies but %s is not a directory'%subpolicy_load_dir

				# load submodel
				node_parent = node[1]['parent']
				is_parent_not_root = ((node_parent.shape[0]>1) or (not (node_parent[0]==-1)))
				file, next_load_iter_ = find_next_load_file(subpolicy_load_dir, current_load_iter, return_max=is_parent_not_root)
				next_load_iter = current_load_iter if is_parent_not_root else max(next_load_iter_, current_load_iter)
				# print('inputs :', active_inputs, 'parent :', node_parent, ', is parent root? :', (not is_parent_not_root), ', load iter :', next_load_iter)
				cummulative_steps += next_load_iter_
				submodel = algorithm.load(file)

				# Update the parent in the input_tree and save subpolicies for reuse
				parent_children = curr_input_tree[inode_parent][1]['children']
				# remove the current node from the parent_children list and add subpolicy weights
				inode_parent_children = np.nonzero([np.all(np.any(pc[:,np.newaxis]==active_inputs, axis=1)) for pc in parent_children])[0]
				assert inode_parent_children.shape[0]==1, 'current node must correspond to a single child in the parent node'

				parent_children.pop(inode_parent_children[0])
				curr_input_tree[inode_parent][1]['children'] = parent_children
				curr_input_tree[inode_parent][1]['state'] = np.logical_or(curr_input_tree[inode_parent][1]['state'], node[1]['state']).astype(np.int32)

				inode_parent_subpolicies = curr_input_tree[inode_parent][1].get('sub_policies', [])
				inode_parent_subpolicies += [active_inputs]
				inode_parent_subpolicies += node[1].get('sub_policies', [])
				curr_input_tree[inode_parent][1]['sub_policies'] = inode_parent_subpolicies

				trained_weights = submodel.policy.get_weights()
				curr_input_tree[inode][1]['weights'] = dict() # save weights corresponding to the active module
				curr_input_tree[inode][1]['weights']['policy_net'] = []
				curr_input_tree[inode][1]['weights']['action_net'] = []
				for imma, mma in enumerate(node_net_arch):
					if (np.all(np.any(active_inputs[:,np.newaxis] == np.array(mma[0]), axis=0))):
						curr_input_tree[inode][1]['weights']['policy_net'] += [[np.array(mma[0]), [(ww[0], ww[1].clone()) for ww in trained_weights['policy_net'][imma]]]]
						curr_input_tree[inode][1]['weights']['action_net'] += [[np.array(mma[0]), [(ww[0], ww[1].clone()) for ww in trained_weights['action_net'][imma]]]]

				del submodel

			else:
				previous_load_iter = current_load_iter
				current_load_iter = next_load_iter
				policy_args['policy_kwargs']['net_arch'] = dict(pi=node_net_arch, vf=net_arch['vf'])
				# create the model
				model = algorithm('DecompositionMlpPolicy', env, **algorithm_args.get('algorithm_kwargs'), policy_kwargs=policy_args.get('policy_kwargs'))

				# initialize frozen model weights
				node_weights = model.policy.get_weights()
				weights_load_dict = dict()
				for imma, mma in enumerate(node_net_arch):
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

				# evaluate model
				print('checkpoint : %d'%current_load_iter)
				episode_rewards, episode_discounted_rewards, final_errors = evaluate_policy(env, model, starts)

				mean_rewards += [np.mean(episode_rewards)]
				mean_discounted_rewards += [np.mean(episode_discounted_rewards)]
				mean_final_errors += [np.mean(final_errors)]
				checkpoint_steps += [deepcopy(cummulative_steps)]
				cummulative_steps = 0
	
	return mean_rewards, mean_discounted_rewards, mean_final_errors, checkpoint_steps


def evaluate_policy(test_env, model, starts, verbose=False):

	num_episodes = starts.shape[0]
	episode_rewards = np.zeros(num_episodes)
	episode_discounted_rewards = np.zeros(num_episodes)
	final_errors = np.zeros(num_episodes)

	for ee in range(num_episodes):
		obs, _ = test_env.reset(state=starts[ee])
		start = test_env.get_obs(normalized=False)
		done = False
		discount = 1
		while (not done):
			action, _state = model.predict(obs, deterministic=True)
			obs, reward, done, _, info = test_env.step(action)

			episode_rewards[ee] += reward
			episode_discounted_rewards[ee] += (discount*reward)
			discount *= model.gamma

		end = test_env.get_obs(normalized=False)
		final_errors[ee] = test_env.get_goal_dist()

		if (verbose):
			with np.printoptions(precision=3, suppress=True):
				print('start obs :', start)
				print('final obs :', end)
				print('reward (discounted) : %f (%f)' %(episode_rewards[ee], episode_discounted_rewards[ee]))
				print('final error : %f' %final_errors[ee])

	return episode_rewards, episode_discounted_rewards, final_errors


def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--load_dirs', nargs='+', default=[], help='list of directories to load the models from')
	parser.add_argument('--labels', nargs='+', default=[], help='list of labels for evaluation plots')
	parser.add_argument('--test_starts', type=str, default='', help='npy file with start configs to test from')
	# parser.add_argument('--num_rollouts', type=int, default=10, help='number of episodes to rollout')
	# parser.add_argument('--record', default=False, action='store_true', help='record the rollouts?')
	# parser.add_argument('--evaluate_lqr', default=False, action='store_true', help='compare against an lqr controller?')

	args = parser.parse_args()
	load_dirs = args.load_dirs
	labels = args.labels
	if (len(labels) < len(load_dirs)):
		len_l = len(labels)
		len_d = len(load_dirs)
		for ii in range(len_l ,len_d):
			labels += os.path.basename(load_dirs[ii])

	elif (len(labels) > len(load_dirs)):
		labels = labels[:len(load_dirs)]

	starts = args.test_starts

	for load_dir, label in zip(load_dirs, labels):
		print('*** Testing %s ***'%label)
		mean_rewards, mean_discounted_rewards, mean_final_errors, checkpoint_steps = evaluate_decomposition_policies(load_dir, test_starts=starts)
		if (not os.path.isfile(starts)) or (not (os.path.splitext(starts)[-1]=='.npy')):
			starts = os.path.join(load_dir, 'test_starts.npy')

		np.savez(
			os.path.join(load_dir, 'summary.npz'), 
			mean_rewards=mean_rewards, 
			mean_discounted_rewards=mean_discounted_rewards, 
			mean_final_errors=mean_final_errors, 
			checkpoint_steps=checkpoint_steps
		)
		plt.plot(checkpoint_steps, mean_discounted_rewards, label=label)

	plt.legend()
	plt.show()

if __name__=='__main__':
	main()