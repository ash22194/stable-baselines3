import argparse
import numpy as np
import torch as th
import gymnasium as gym

from stable_baselines3.gpu_systems import GPUQuadcopter, GPUUnicycle

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='quadcopter', help='environemnt to test')
	parser.add_argument('--device', type=str, default='cuda', help='cuda/cpu')

	normalized_observations = False
	args = parser.parse_args()
	env_name = args.env_name
	device = args.device
	num_envs = 100
	
	th.set_default_tensor_type(th.DoubleTensor)

	if (env_name == 'quadcopter'):
		env_cpu = gym.make('Quadcopter-v0', normalized_observations=normalized_observations)
		env_gpu = GPUQuadcopter(device=device, num_envs=num_envs, normalized_observations=normalized_observations)
	elif (env_name == 'unicycle'):
		env_cpu = gym.make('Unicycle-v0', normalized_observations=normalized_observations)
		env_gpu = GPUUnicycle(device=device, num_envs=num_envs, normalized_observations=normalized_observations)
	else:
		NotImplementedError

	start_cpu = th.cuda.Event(enable_timing=True)
	end_cpu = th.cuda.Event(enable_timing=True)
	start_gpu = th.cuda.Event(enable_timing=True)
	end_gpu = th.cuda.Event(enable_timing=True)
	
	num_samples = 1000
	dyn_err = []
	# Test dyn first
	for nn in range(num_samples):
		obs_cpu, _ = env_cpu.reset()
		obs_gpu, _ = env_gpu.reset(state=(th.ones((num_envs, 1), device=device) * th.asarray(env_cpu.state, device=device)))

		action = env_cpu.action_space.low + np.random.rand()*(env_cpu.action_space.high - env_cpu.action_space.low)

		dx_cpu = env_cpu.dyn_full(env_cpu.state[:,np.newaxis], action[:,np.newaxis])
		dx_gpu = env_gpu.dyn_full(env_gpu.state, th.asarray(action[np.newaxis,:], device=device))

		dyn_err += [np.max(np.abs(dx_cpu - (dx_gpu[0:1,:].cpu().clone().numpy().T)))]
	
	print("Mean (max) max dyn err : %.6f (%.6f)"%(np.mean(np.array(dyn_err)), np.max(np.array(dyn_err))))

	done = False
	obs_cpu, _ = env_cpu.reset()
	obs_gpu, _ = env_gpu.reset(state=(th.ones((num_envs, 1), device=device) * th.asarray(env_cpu.state, device=device)))
	
	action_cpu = []
	dx_cpu = []
	dx_gpu = []
	traj_obs_cpu = [obs_cpu]
	traj_obs_gpu = [obs_gpu.cpu().clone().numpy()]
	traj_rew_cpu = []
	traj_rew_gpu = []
	step_time_cpu = []
	step_time_gpu = []
	count = 0
	while (not done):
		action = env_cpu.action_space.low + np.random.rand()*(env_cpu.action_space.high - env_cpu.action_space.low)
		action_cpu += [action.copy()]

		# compute accelerations
		dx_cpu += [env_cpu.dyn_full(env_cpu.state[:,np.newaxis], action[:,np.newaxis])]
		dx_gpu += [env_gpu.dyn_full(env_gpu.state, th.asarray(action[np.newaxis,:], device=device))] 

		start_cpu.record()
		obs_cpu, rew_cpu, done, _, _ = env_cpu.step(action)
		end_cpu.record()

		start_gpu.record()
		obs_gpu, rew_gpu, done_gpu, _, _ = env_gpu.step(th.asarray(action[np.newaxis,:], device=device))
		end_gpu.record()

		th.cuda.synchronize()

		traj_obs_cpu += [obs_cpu]
		traj_obs_gpu += [obs_gpu.cpu().clone().numpy()]

		traj_rew_cpu += [rew_cpu]
		traj_rew_gpu += [rew_gpu.cpu().clone().numpy()]

		step_time_cpu += [start_cpu.elapsed_time(end_cpu)]
		step_time_gpu += [start_gpu.elapsed_time(end_gpu)]

		print('Current step time gpu (cpu) %d : %.6f (%.6f)'%(count, step_time_gpu[-1], step_time_cpu[-1]))
		count += 1

	traj_obs_gpu_first = [ob[0,:] for ob in traj_obs_gpu]
	traj_rew_gpu_first = [r for r in traj_rew_gpu]

	obs_err = np.array([np.max(np.abs(traj_obs_cpu[ii] - traj_obs_gpu_first[ii])) for ii in range(len(traj_obs_gpu))])
	rew_err = np.array([np.max(np.abs(traj_rew_cpu[ii] - traj_rew_gpu_first[ii])) for ii in range(len(traj_rew_gpu))])

	print('Mean max obs error : %.6f', np.mean(obs_err))
	print('Mean reward error : %.6f', np.mean(rew_err))

if __name__=='__main__':
	main()