import torch
from torch import nn
import os
import argparse
from shutil import copyfile
from ruamel.yaml import YAML
import gymnasium as gym
import numpy as np
from copy import deepcopy

from stable_baselines3 import A2CwReg, PPO
from stable_baselines3.common.env_checker import check_env

def evaluate_model(model, test_env: gym.Env, num_episodes: int, print_outcomes=False):
	ep_reward = np.zeros(num_episodes)
	ep_discounted_reward = np.zeros(num_episodes)
	final_err = np.zeros(num_episodes)

	for ee in range(num_episodes):
		obs, _ = test_env.reset()
		start = deepcopy(obs)
		done = False
		discount = 1
		while (not done):
			action, _state = model.predict(obs, deterministic=True)
			obs, reward, done, _, info = test_env.step(action)
			test_env.render()

			ep_reward[ee] += reward
			ep_discounted_reward[ee] += (discount*reward)
			discount *= model.gamma

		final_err[ee] = np.linalg.norm(obs.reshape(-1) - test_env.goal.reshape(-1))

		if (print_outcomes):
			with np.printoptions(precision=3, suppress=True):
				print('Episode %d :' % ee)
				print('---Start state : ', start)
				print('---End state : ', obs)
				print('---Reward (discounted) : %f (%f)' % (ep_reward[ee], ep_discounted_reward[ee]))

	return ep_reward, ep_discounted_reward, final_err

def main():
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--load_dir', type=str, default='', help='directory to load the model from')

	args = parser.parse_args()
	load_dir = args.load_dir
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
	
	test_env = gym.make(cfg['environment']['name'], **cfg['environment']['environment_kwargs'])
	check_env(test_env)

	if (cfg['algorithm']['name'] == 'PPO'):
		model = PPO.load(ff_load)
	elif (cfg['algorithm']['name'] == 'A2C'):
		pass
	
	evaluate_model(model, test_env, num_episodes=5, print_outcomes=True)

if __name__=='__main__':
	main()