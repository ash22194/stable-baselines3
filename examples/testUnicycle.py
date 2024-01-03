
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from copy import deepcopy
import gymnasium as gym
from cProfile import Profile
import time
from ipdb import set_trace

from stable_baselines3.common.env_checker import check_env

def compute_taskspace_jacobian(env, x):

	J = np.zeros((8, env.X_DIMS))
	eps = 1e-6
	for idd in env.independent_dims:
		ev = np.zeros(env.X_DIMS)
		ev[idd] = eps
		xp = x + ev
		env.reset(state=xp)
		yp = env.get_taskspace_obs()

		xm = x - ev
		env.reset(state=xm)
		ym = env.get_taskspace_obs()

		J[:, idd] = (yp - ym) / (2*eps)

	return J

def compute_taskspace_hessian(env, x):

	H = zeros((8, env.X_DIMS, env.X_DIMS))
	eps = 1e-6
	for idd in env.independent_dims:
		ev = np.zeros(env.X_DIMS)
		ev[idd] = eps

		xp = x + ev
		Jp = compute_taskspace_jacobian(env, xp)

		xm = x - ev
		Jm = compute_taskspace_jacobian(env, xm)

		H[:,:,idd] = (Jp - Jm) / (2*eps)

	return H

def main():
	normalized_actions = True
	fixed_start = True
	env_kwargs = dict(fixed_start=fixed_start, normalized_actions=normalized_actions)
	env = gym.make('Unicycle-v0', **env_kwargs)
	# check_env(env)

	# Test dynamics
	prof = Profile()
	prof.enable()
	num_samples = 100
	dx_errs = np.zeros(num_samples)
	dyn_time = 0
	dyn_ana_time = 0
	for nn in range(num_samples):
		env.reset()
		state = env.state
		inp = 0.5 * (env.u_limits[:,1] + env.u_limits[:,0]) + (np.random.rand(env.U_DIMS) - 0.5) * (env.u_limits[:,1] - env.u_limits[:,0])
		start_dyn = time.process_time()
		dx = env.dyn_full(state[:,np.newaxis], inp[:,np.newaxis])
		dyn_time += (time.process_time() - start_dyn)

		start_dyn_ana = time.process_time()
		dx_ana = env.dyn_full_ana(state[:,np.newaxis], inp[:,np.newaxis])
		dyn_ana_time += (time.process_time() - start_dyn_ana)

		dx_errs[nn] = np.linalg.norm(dx - dx_ana)
	prof.disable()
	print('Dynamics discrepancy :', dx_errs)
	print('Dyn time :', dyn_time / num_samples)
	print('Dyn ana time :', dyn_ana_time / num_samples)

	# prof.create_stats()
	# prof.dump_stats('unicycle_profile')
	env.reset(state=env.goal[:,0])
	goal = env.state[:,np.newaxis]
	x0 = goal
	u0 = env.u0
	A = np.zeros((env.X_DIMS, env.X_DIMS))
	B = np.zeros((env.X_DIMS, env.U_DIMS))
	eps = 1e-6
	for xx in range(env.X_DIMS):
		perturb = np.zeros((env.X_DIMS, 1))
		perturb[xx, 0] = eps
		dyn_p = env.dyn_full(x0 + perturb, u0)
		dyn_m = env.dyn_full(x0 - perturb, u0)

		A[:, xx:(xx+1)] = (dyn_p - dyn_m) / (2*eps)

	for uu in range(env.U_DIMS):
		perturb = np.zeros(u0.shape)
		perturb[uu] = eps
		dyn_p = env.dyn_full(x0, u0 + perturb)
		dyn_m = env.dyn_full(x0, u0 - perturb)

		B[:, uu:(uu+1)] = (dyn_p - dyn_m) / (2*eps)

	J = compute_taskspace_jacobian(env, goal[:,0])
	Q = (J.T) @ env.Q @ J
	R = env.R
	lambda_ = 0.5
	gamma_ = 0.9995
	P = solve_continuous_are(A - lambda_/2*np.eye(A.shape[0]), B, Q, R)
	K = np.matmul(np.linalg.inv(R), np.matmul(B.T, P))

	with np.printoptions(precision=3, suppress=True):
		print('A :', A)
		print('B :', B)
		print('Q :', Q)
		print('R :', R)
		print('LQR gain :', K)

	# Test the policy
	err_p = np.zeros((3, int(env.T / env.dt)))
	err_v = np.zeros((3, int(env.T / env.dt)))
	err_a = np.zeros((3, int(env.T / env.dt)))
	obs, info = env.reset()
	env.render()
	set_trace()
	start = env.state
	u_limits = env.u_limits
	value = 0
	discount = 1
	for i in range(int(env.T / env.dt)):
		action = np.matmul(-K, env.state[:,np.newaxis] - goal)[:,0]
		action = np.maximum(u_limits[:,0], np.minimum(u_limits[:,1], action))
		with np.printoptions(precision=3, suppress=True):
			print('u :', action)
			print('v :', env.state[8:])

		if (normalized_actions):
			action = (2*action - (u_limits[:,1] + u_limits[:,0])) / (u_limits[:,1] - u_limits[:,0])

		with np.printoptions(precision=3, suppress=True):
			print('dv :', env.dyn_full(env.state[:,np.newaxis], action[:,np.newaxis])[8:,0])

		obs, reward, done, _, info = env.step(action)
		env.render()
		err_p[:,i] = deepcopy(info['contact_info']['err_p'])
		err_v[:,i] = deepcopy(info['contact_info']['err_v'])
		err_a[:,i] = deepcopy(info['contact_info']['err_a'])
		with np.printoptions(precision=3, suppress=True):
			print('e_p :', err_p[:,i])
			print('e_v :', err_v[:,i])
			print('e_a :', err_a[:,i])

		value += (discount*reward)
		discount *= gamma_
		# set_trace()
		if done:
			err_p = err_p[:,:i]
			err_v = err_v[:,:i]
			err_a = err_a[:,:i]
			print('Start state :', start)
			print('Final state :', env.state)
			print('Value :', value)
			break

	fig, ax = plt.subplots(3)
	time_steps = np.linspace(0, i*env.dt, i)
	ax[0].plot(time_steps, err_p[0,:], color='r')
	ax[0].plot(time_steps, err_p[1,:], color='g')
	ax[0].plot(time_steps, err_p[2,:], color='b')

	ax[1].plot(time_steps, err_v[0,:], color='r')
	ax[1].plot(time_steps, err_v[1,:], color='g')
	ax[1].plot(time_steps, err_v[2,:], color='b')

	ax[2].plot(time_steps, err_a[0,:], color='r')
	ax[2].plot(time_steps, err_a[1,:], color='g')
	ax[2].plot(time_steps, err_a[2,:], color='b')

	plt.show()

	set_trace()
if __name__=='__main__':
	main()