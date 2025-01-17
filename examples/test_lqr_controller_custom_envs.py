import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are, expm
import gymnasium as gym
import argparse

import stable_baselines3.systems

def compute_dynamics_linearizations(env):

	env.reset()
	x0 = np.zeros((env.state.shape[0], 1))
	x0[env.observation_dims,0] = env.goal[:,0]
	u0 = env.u0 
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

		A[:,xx:(xx+1)] = (env.dyn_full(xp, u0) - env.dyn_full(xm, u0)) / (2*eps)

	for uu in range(nu):
		eu = np.zeros((nu, 1))
		eu[uu] = eps
		up = u0 + eu
		um = u0 - eu

		B[:,uu:(uu+1)] = (env.dyn_full(x0, up) - env.dyn_full(x0, um)) / (2*eps)

	A = A[env.observation_dims,:]
	A = A[:,env.observation_dims]
	B = B[env.observation_dims,:]
	
	Ad = expm(A*env.dt)
	Bd = B*env.dt

	return A, B, Ad, Bd

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='linearsystem', help='environment name')
	parser.add_argument('--controller_type', type=str, default='ct', help='ct or dt continuous or discrete time')
	args = parser.parse_args()

	if (args.env_name == 'quadcopter'):
		env = gym.make('Quadcopter-v0', normalized_observations=False)

	elif (args.env_name == 'unicycle'):
		env = gym.make('Unicycle-v0', fixed_start=True, normalized_observations=False)

	# compute linearizations
	A, B, Ad, Bd = compute_dynamics_linearizations(env)

	# compute LQR controller
	if (args.controller_type == 'ct'):
		# continuous time
		lambda_ = env.lambda_
		P = solve_continuous_are(A - lambda_/2*np.eye(A.shape[0]), B, env.Q, env.R)
		K = np.linalg.inv(env.R) @ (B.T @ P)
	elif (args.controller_type == 'dt'):
		# discrete time
		Qd = env.Q*env.dt
		Rd = env.R*env.dt
		gamma_ = env.gamma_
		P = solve_discrete_are(Ad*np.sqrt(gamma_), Bd, Qd, Rd/gamma_)
		K = gamma_ * np.linalg.inv(Rd + gamma_*(Bd.T @ P) @ Bd) @ (Bd.T @ P) @ Ad

	num_episodes = 5
	u0 = env.u0[:,0]
	goal = env.goal[:,0]
	normalized_actions = env.normalized_actions
	u_limits = env.u_limits

	for nn in range(num_episodes):
		_, _ = env.reset()
		env.render()
		obs = env.get_obs(normalized=False)
		start = deepcopy(obs)
		done = False
		value = 0
		min_obs = np.inf*np.ones(obs.shape)
		max_obs = -np.inf*np.ones(obs.shape)
		while (not done):
			action = (u0 - K @ (obs - goal))
			action = np.maximum(u_limits[:,0], np.minimum(u_limits[:,1], action))
			if (normalized_actions):
				action = (2*action - (u_limits[:,1] + u_limits[:,0])) / (u_limits[:,1] - u_limits[:,0])
			
			_, reward, done, _, info = env.step(action)
			obs = env.get_obs(normalized=False)
			env.render()
			min_obs[obs < min_obs] = obs[obs < min_obs]
			max_obs[obs > max_obs] = obs[obs > max_obs]
			value += reward

		with np.printoptions(precision=3, suppress=True):
			print('Start state :', start, ', Final state :', obs, ', Value :', value, ', Step count :', info.get('step_count'))
			print('Min obs :', min_obs)
			print('Max obs :', max_obs)
			print()

if __name__=='__main__':
	main()