
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from copy import deepcopy
import gymnasium as gym

from stable_baselines3.common.env_checker import check_env

def main():
	normalized_actions = True
	fixed_start = False
	expand_limits = False
	env_kwargs = dict(fixed_start=fixed_start, normalized_actions=normalized_actions, expand_limits=expand_limits)
	env = gym.make('Unicycle-v0', **env_kwargs)
	check_env(env)

	goal = env.goal
	u0 = env.u0
	A = np.zeros((env.X_DIMS, env.X_DIMS))
	B = np.zeros((env.X_DIMS, env.U_DIMS))
	eps = 1e-6
	for xx in range(env.X_DIMS):
		perturb_p = np.zeros(goal.shape)
		perturb_p[xx] = eps
		perturb_m = np.zeros(goal.shape)
		perturb_m[xx] = -eps
		dyn_p = env.dyn(goal + perturb_p, u0)
		dyn_m = env.dyn(goal + perturb_m, u0)

		A[:, xx:(xx+1)] = (dyn_p - dyn_m) / (2*eps)
	
	for uu in range(env.U_DIMS):
		perturb_p = np.zeros(u0.shape)
		perturb_p[uu] = eps
		perturb_m = np.zeros(u0.shape)
		perturb_m[uu] = -eps
		dyn_p = env.dyn(goal, u0 + perturb_p)
		dyn_m = env.dyn(goal, u0 + perturb_m)

		B[:, uu:(uu+1)] = (dyn_p - dyn_m) / (2*eps)

	Q = env.Q
	R = env.R
	lambda_ = 0.5
	gamma_ = 0.9995
	P = solve_continuous_are(A - lambda_/2*np.eye(env.X_DIMS), B, Q, R)
	K = np.matmul(np.linalg.inv(R), np.matmul(B.T, P))

	with np.printoptions(precision=3, suppress=True):
		print('A :', A)
		print('B :', B)
		print('Q :', Q)
		print('R :', R)
		print('LQR gain :', K)

	# Test the policy
	obs, _ = env.reset()
	start = obs
	u_limits = env.u_limits
	value = 0
	discount = 1
	for i in range(int(env.T / env.dt)):
		action = np.matmul(-K, obs[:,np.newaxis] - goal)[:,0]
		action = np.maximum(u_limits[:,0], np.minimum(u_limits[:,1], action))
		if (normalized_actions):
			action = (2*action - (u_limits[:,1] + u_limits[:,0])) / (u_limits[:,1] - u_limits[:,0])

		obs, reward, done, _, info = env.step(action)
		value += (discount*reward)
		discount *= gamma_

		if done:
			print('Start state :', start, ', Final state :', obs, 'Value :', value)
			obs, _ = env.reset()
			start = obs
			value = 0

if __name__=='__main__':
	main()