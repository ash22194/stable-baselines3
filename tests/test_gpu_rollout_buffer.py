import argparse

import torch as th
import numpy as np

from stable_baselines3.gpu_systems import GPUQuadcopter, GPUUnicycle
from stable_baselines3.common.vec_env import GPUVecEnv
from stable_baselines3.common.buffers import GPURolloutBuffer,RolloutBuffer
from stable_baselines3.common.env_util import make_vec_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='quadcopter', help='environment to test with')
    parser.add_argument('--num_envs', type=int, default=100, help='number of parallel environments to create')
    parser.add_argument('--n_steps', type=int, default=100, help='number of parallel environments to create')
    parser.add_argument('--buffer_size', type=int, default=100, help='length of the buffer')

    args = parser.parse_args()
    th.set_default_tensor_type(th.FloatTensor)

    env_name = args.env_name
    num_envs = args.num_envs
    n_steps = args.n_steps
    batch_size = n_steps*num_envs
    buffer_size = args.buffer_size
    device = 'cuda'
    if (env_name=='quadcopter'):
        env_gpu = GPUVecEnv(GPUQuadcopter(device=device), num_envs=num_envs)
    elif (env_name=='unicycle'):
        env_gpu = GPUVecEnv(GPUUnicycle(device=device), num_envs=num_envs)
    else:
        NotImplementedError

    buffer_gpu = GPURolloutBuffer(
        buffer_size=buffer_size,
        observation_space=env_gpu.observation_space,
        action_space=env_gpu.action_space,
        device=device,
        n_envs=num_envs
    )
    buffer_cpu = RolloutBuffer(
        buffer_size=buffer_size,
        observation_space=env_gpu.observation_space,
        action_space=env_gpu.action_space,
        n_envs=num_envs
    )

    last_obs_gpu = env_gpu.reset()
    action_gpu = th.zeros((num_envs, env_gpu.action_space.shape[0]), device=device)
    rew_gpu = th.zeros((num_envs,), device=device)
    last_episode_starts_gpu = th.ones((num_envs,), dtype=bool, device=device)
    
    last_obs_cpu = last_obs_gpu.cpu().numpy()
    action_cpu = action_gpu.cpu().numpy()
    rew_cpu = rew_gpu.cpu().numpy()
    last_episode_starts_cpu = last_episode_starts_gpu.cpu().numpy()

    step_count = 0
    done = th.zeros((num_envs,), dtype=bool, device=device)
    while (not th.all(done)):
        
        value = th.rand(num_envs, device=device)
        log_prob = -th.rand(num_envs, device=device)
        action_gpu = th.asarray(env_gpu.action_space.low + np.random.rand(num_envs, env_gpu.action_space.shape[0])*(env_gpu.action_space.high - env_gpu.action_space.low),
            device=device
        )
        new_obs_gpu, rew_gpu, done, info_gpu = env_gpu.step(action_gpu)
        buffer_gpu.add(last_obs_gpu, action_gpu, rew_gpu, last_episode_starts_gpu, value, log_prob)
        
        last_obs_cpu = last_obs_gpu.cpu().numpy()
        action_cpu = action_gpu.cpu().numpy()
        rew_cpu = rew_gpu.cpu().numpy()
        last_episode_starts_cpu = last_episode_starts_gpu.cpu().numpy()
        buffer_cpu.add(last_obs_cpu, action_cpu, rew_cpu, last_episode_starts_cpu, value, log_prob)
        
        last_obs_gpu = new_obs_gpu
        last_episode_starts_gpu = done

        step_count += 1

        if (step_count % n_steps) == 0:
            last_values = th.rand(num_envs, device=device)
            buffer_gpu.compute_returns_and_advantage(last_values=last_values, dones=done)
            buffer_cpu.compute_returns_and_advantage(last_values=last_values, dones=done.cpu().numpy())

            for batch_gpu, batch_cpu in zip(buffer_gpu.get(batch_size), buffer_cpu.get(batch_size)):
                # What quantities to check?
                max_diff_actions = th.max(th.abs(batch_gpu.actions - batch_cpu.actions))
                max_diff_observations = th.max(th.abs(batch_gpu.observations - batch_cpu.observations))
                max_diff_advantages = th.max(th.abs(batch_gpu.advantages - batch_cpu.advantages))
                max_diff_old_log_prob = th.max(th.abs(batch_gpu.old_log_prob - batch_cpu.old_log_prob))
                max_diff_old_values = th.max(th.abs(batch_gpu.old_values - batch_cpu.old_values))
                max_diff_returns = th.max(th.abs(batch_gpu.returns - batch_cpu.returns))

                print('step count :', step_count)
                print('max diff actions', max_diff_actions)
                print('max diff observations', max_diff_observations)
                print('max diff advantages', max_diff_advantages)
                print('max diff old_log_prob', max_diff_old_log_prob)
                print('max diff old_values', max_diff_old_values)
                print('max diff returns', max_diff_returns)
                print(' ')

                err_thresh = 1e-4
                if (
                    (max_diff_actions > err_thresh) or
                    (max_diff_observations > err_thresh) or
                    (max_diff_advantages > err_thresh) or
                    (max_diff_old_log_prob > err_thresh) or
                    (max_diff_old_values > err_thresh) or
                    (max_diff_returns > err_thresh)
                ):
                    pass # If need to check
            
            buffer_gpu.reset()
            buffer_cpu.reset()

if __name__=='__main__':
    main()