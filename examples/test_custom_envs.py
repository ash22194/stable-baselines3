import torch
from torch import nn
import os
import argparse
from shutil import copyfile
from ruamel.yaml import YAML
import gymnasium as gym
import numpy as np
from copy import deepcopy
from scipy.linalg import solve_discrete_are, expm

from stable_baselines3 import A2CwReg, PPO
from stable_baselines3.common.env_checker import check_env

def evaluate_model(model, test_env: gym.Env, num_episodes: int, print_outcomes=False, record=False):
	ep_reward = np.zeros(num_episodes)
	ep_discounted_reward = np.zeros(num_episodes)
	final_err = np.zeros(num_episodes)
	start_states = np.zeros((test_env.X_DIMS, num_episodes))

	for ee in range(num_episodes):
		if (os.path.isdir(record)):
			current_record_dir = os.path.join(record, str(ee))
			os.makedirs(current_record_dir, exist_ok=True)
		obs, _ = test_env.reset()
		start_states[:,ee] = test_env.state
		start_obs = test_env.get_obs(normalized=False)
		done = False
		discount = 1
		step_count = 0
		while (not done):
			action, _state = model.predict(obs, deterministic=True)
			obs, reward, done, _, info = test_env.step(action)
			im = test_env.render()
			if (os.path.isdir(record)):
				im.save(os.path.join(current_record_dir, '%04d.png'%step_count))

			ep_reward[ee] += reward
			ep_discounted_reward[ee] += (discount*reward)
			discount *= model.gamma
			step_count += 1

		end_obs = test_env.get_obs(normalized=False)
		final_err[ee] = test_env.get_goal_dist()

		if (print_outcomes):
			with np.printoptions(precision=3, suppress=True):
				print('Episode %d :' % ee)
				print('---Start obs : ', start_obs)
				print('---End obs : ', end_obs)
				print('---Reward (discounted) : %f (%f)' % (ep_reward[ee], ep_discounted_reward[ee]))
				print('---Final : %f' % final_err[ee])
				print()

	return ep_reward, ep_discounted_reward, final_err, start_states

def evaluate_lqr_controller(starts, test_env: gym.Env, num_episodes: int, print_outcomes=False, record=False):
	
	assert starts.shape[1]==num_episodes, 'Number of starts and number of episodes do not match'

	test_env.reset()
	x0 = np.zeros((test_env.state.shape[0], 1))
	x0[test_env.observation_dims,0] = test_env.goal[:,0]
	goal = test_env.goal
	u0 = test_env.u0 
	nx = x0.shape[0]
	nu = u0.shape[0]

	A = np.zeros((nx, nx))
	B = np.zeros((nx, nu))
	eps = 1e-6
	for xx in range(nx):
		ev = np.zeros((nx, 1))
		ev[xx,0] = eps
		xp = x0 + ev
		xm = x0 - ev

		A[:,xx:(xx+1)] = (test_env.dyn_full(xp, u0) - test_env.dyn_full(xm, u0)) / (2*eps)

	for uu in range(nu):
		eu = np.zeros((nu, 1))
		eu[uu] = eps
		up = u0 + eu
		um = u0 - eu

		B[:,uu:(uu+1)] = (test_env.dyn_full(x0, up) - test_env.dyn_full(x0, um)) / (2*eps)

	A = A[test_env.observation_dims,:]
	A = A[:,test_env.observation_dims]
	B = B[test_env.observation_dims,:]
	
	Ad = expm(A*test_env.dt)
	Bd = B*test_env.dt

	# discrete time
	Qd = test_env.Q*test_env.dt
	Rd = test_env.R*test_env.dt
	gamma_ = test_env.gamma_
	P = solve_discrete_are(Ad*np.sqrt(gamma_), Bd, Qd, Rd/gamma_)
	K = gamma_ * np.linalg.inv(Rd + gamma_*(Bd.T @ P) @ Bd) @ (Bd.T @ P) @ Ad
	goal = goal[:,0]
	u0 = u0[:,0]
	u_limits = test_env.u_limits
	normalized_actions = test_env.normalized_actions

	ep_reward = np.zeros(num_episodes)
	ep_discounted_reward = np.zeros(num_episodes)
	final_err = np.zeros(num_episodes)
	for ee in range(num_episodes):
		if (os.path.isdir(record)):
			current_record_dir = os.path.join(record, 'lqr_%d'%ee)
			os.makedirs(current_record_dir, exist_ok=True)
		_, _ = test_env.reset(state=starts[:,ee])
		obs = test_env.get_obs(normalized=False)
		start_obs = deepcopy(obs)
		done = False
		discount = 1
		step_count = 0
		while (not done):
			action = (u0 - K @ (obs[:test_env.observation_dims.shape[0]] - goal))
			action = np.maximum(u_limits[:,0], np.minimum(u_limits[:,1], action))
			if (normalized_actions):
				action = (2*action - (u_limits[:,1] + u_limits[:,0])) / (u_limits[:,1] - u_limits[:,0])

			_, reward, done, _, info = test_env.step(action)
			obs = test_env.get_obs(normalized=False)
			im = test_env.render()
			if (os.path.isdir(record)):
				im.save(os.path.join(current_record_dir, '%04d.png'%step_count))

			ep_reward[ee] += reward
			ep_discounted_reward[ee] += (discount*reward)
			discount *= gamma_
			step_count += 1

		end_obs = test_env.get_obs(normalized=False)
		final_err[ee] = test_env.get_goal_dist()

		if (print_outcomes):
			with np.printoptions(precision=3, suppress=True):
				print('Episode %d :' % ee)
				print('---Start obs : ', start_obs)
				print('---End obs : ', end_obs)
				print('---Reward (discounted) : %f (%f)' % (ep_reward[ee], ep_discounted_reward[ee]))
				print('---Final : %f' % final_err[ee])
				print()

	return ep_reward, ep_discounted_reward, final_err

def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--load_dir', type=str, default='', help='directory to load the model from')
	parser.add_argument('--record', default=False, action='store_true', help='record the rollouts?')
	parser.add_argument('--evaluate_lqr', default=False, action='store_true', help='compare against an lqr controller?')

	args = parser.parse_args()
	load_dir = args.load_dir
	record = args.record
	if (record):
		record = load_dir
	files = [os.path.join(load_dir, f) for f in os.listdir(load_dir) if os.path.isfile(os.path.join(load_dir, f))]

	zip_files = []
	curr_save_id = 0
	# find the cfg file
	for ff in files:
		ff_basename = os.path.basename(ff)
		ff_name, ff_ext = os.path.splitext(ff_basename)
		if (ff_ext == '.zip'):
			ff_save_id = int(ff_name.split('_')[-1])
			if (ff_save_id > curr_save_id):
				curr_save_id = ff_save_id
				ff_load = ff

		elif (ff_ext == '.yaml'):
			cfg = YAML().load(open(ff, 'r'))
	
	eval_envname = cfg['environment'].get('eval_envname', cfg['environment']['name'])
	test_env = gym.make(eval_envname, **cfg['environment']['environment_kwargs'])
	check_env(test_env)

	if (cfg['algorithm']['name'] == 'PPO'):
		model = PPO.load(ff_load)
	elif (cfg['algorithm']['name'] == 'A2C'):
		pass
	test_env.gamma_ = model.gamma
	_, _, _, starts = evaluate_model(model, test_env, num_episodes=10, print_outcomes=True, record=record)
	if (args.evaluate_lqr):
		evaluate_lqr_controller(starts, test_env, num_episodes=starts.shape[1], print_outcomes=True, record=record)

if __name__=='__main__':
	main()