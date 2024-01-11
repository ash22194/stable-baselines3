import argparse
import numpy as np
import torch as th
import gymnasium as gym

from stable_baselines3.gpu_systems import GPUQuadcopter
from stable_baselines3.common.vec_env import GPUVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='quadcopter', help='environemnt to test')
	parser.add_argument('--num_envs', type=int, default=100, help='number of instanecs to create')
	args = parser.parse_args()

	env_name = args.env_name
	num_envs = args.num_envs
	normalized_observations = False

	th.set_default_tensor_type(th.DoubleTensor)

	if (env_name=='quadcopter'):
		env_cpu = make_vec_env('Quadcopter-v0', num_envs, env_kwargs=dict(normalized_observations=normalized_observations))
		env_gpu = GPUVecEnv(GPUQuadcopter(device='cuda', normalized_observations=normalized_observations), num_envs=num_envs)
	else:
		NotImplementedError

	goal = th.zeros(12, device='cuda')
	goal[2:] = env_gpu.get_attr('th_goal')[0]+th.rand(10, device='cuda')*0.1
	goal = th.ones((num_envs, 1), device='cuda') * goal
	obs_gpu, _ = env_gpu.env_method('reset')[0]
	state_gpu = env_gpu.get_attr('state')[0].cpu().numpy()
	obs_cpu = np.zeros(obs_gpu.shape)
	for ee in range(num_envs):
		obs_cpu[ee,:] = env_cpu.env_method(method_name='reset', state=state_gpu[ee,:], indices=[ee])[0][0]

	start_cpu = th.cuda.Event(enable_timing=True)
	end_cpu = th.cuda.Event(enable_timing=True)
	start_gpu = th.cuda.Event(enable_timing=True)
	end_gpu = th.cuda.Event(enable_timing=True)
	
	done = False
	traj_obs_cpu = [obs_cpu]
	traj_obs_gpu = [obs_gpu.cpu().numpy()]
	traj_rew_cpu = []
	traj_rew_gpu = []
	step_time_cpu = []
	step_time_gpu = []
	count = 0
	while (not done):
		# action = env_cpu.action_space.low + np.random.rand(num_envs, env_cpu.action_space.low.shape[0])*(env_cpu.action_space.high - env_cpu.action_space.low)
		action = np.ones((num_envs, 1)) @ (env_cpu.action_space.high)[np.newaxis,:]

		start_cpu.record()
		obs_cpu, rew_cpu, done_cpu, info_cpu = env_cpu.step(action)
		end_cpu.record()

		start_gpu.record()
		obs_gpu, rew_gpu, done_gpu, info_gpu = env_gpu.step(th.asarray(action, device='cuda'))
		end_gpu.record()

		th.cuda.synchronize()

		traj_obs_cpu += [obs_cpu]
		traj_obs_gpu += [obs_gpu.cpu().numpy()]

		traj_rew_cpu += [rew_cpu]
		traj_rew_gpu += [rew_gpu.cpu().numpy()]

		step_time_cpu += [start_cpu.elapsed_time(end_cpu)]
		step_time_gpu += [start_gpu.elapsed_time(end_gpu)]

		print('Current step time gpu (cpu) %d : %.6f (%.6f) err %.6f'%(count, step_time_gpu[-1], step_time_cpu[-1], np.max(np.abs(traj_obs_gpu[-1] - traj_obs_cpu[-1]))))
		count += 1
		done = np.all(np.array(done_cpu))
	
	obs_err = np.array([np.max(np.abs(traj_obs_cpu[ii] - traj_obs_gpu[ii])) for ii in range(len(traj_obs_cpu)-1)])
	obs_err_ = [np.max(np.abs(traj_obs_cpu[ii] - traj_obs_gpu[ii]), axis=1) for ii in range(len(traj_obs_cpu)-1)]
	rew_err = np.array([np.max(np.abs(traj_rew_cpu[ii] - traj_rew_gpu[ii])) for ii in range(len(traj_rew_cpu)-1)])
	step_time_cpu = np.array(step_time_cpu)
	step_time_gpu = np.array(step_time_gpu)

	print('Mean max err obs (rew) : %.6f (%.6f)'%(np.mean(obs_err), np.mean(rew_err)))
	print('Step time gpu (cpu) : %.6f (%.6f)'%(np.mean(step_time_gpu), np.mean(step_time_cpu)))

if __name__=='__main__':
	main()