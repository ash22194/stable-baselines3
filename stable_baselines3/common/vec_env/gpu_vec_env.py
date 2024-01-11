from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union

import gymnasium as gym
import torch as th

from stable_baselines3.common.vec_env.base_vec_env import VecEnv

class GPUVecEnv(VecEnv):
    """
    Creates a wrapper for environments that accept batch of actions and return batch of states, 
    steps are run in a batch fashion this is useful for computationally simple environment such as ``Cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that require a vectorized environment, but that you want a single environments to train with.

    :param env: an environment that will accept batch of actions and perform batched steps and resets
    """

    def __init__(self, env, num_envs=1):
        self.envs = [env]
        self.envs[0].set_num_envs(num_envs)
        self.num_envs = num_envs
        self.device = env.device
        self.metadata = env.metadata

        super().__init__(num_envs, env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.buf_obs = th.zeros((1,) + (self.num_envs,) + tuple(obs_space.shape), device=self.device) # No dictionary of obs but a single proprioceptive tensor

        self.action_space_low = th.asarray(self.action_space.low, device=self.device)
        self.action_space_high = th.asarray(self.action_space.high, device=self.device)

        self.buf_dones = th.zeros((self.num_envs,), dtype=th.bool, device=self.device)
        self.buf_rews = th.zeros((self.num_envs,), dtype=th.float32, device=self.device)
        self.buf_infos = [{}]
        self.actions: th.FloatTensor = None
    
    def step_async(self, actions) -> None:
        self.actions = actions
    
    def step_wait(self):
        self.buf_obs[0], self.buf_rews, self.buf_dones, _, self.buf_infos[0] = self.envs[0].step(self.actions)
        if th.any(self.buf_dones):
            self.buf_obs[0] = self.reset()

        return self.buf_obs[0], self.buf_rews, self.buf_dones, self.buf_infos
    
    def reset(self):
        obs, _ = self.envs[0].reset(seed=self._seeds[0], done=self.buf_dones)
        self._reset_seeds()
        return obs
    
    def close(self) -> None:
        for env in self.envs:
            env.close()
    
    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return [None] * self.env_count

    def get_images(self):
        pass
    
    def render(self, mode: Optional[str] = None):
        pass
    
    def _obs_from_buf(self):
        return self.buf_obs

    def get_attr(self, attr_name: str, indices = None) -> List[Any]:
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]
    
    def set_attr(self, attr_name: str, value: Any, indices = None) -> None:
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)
    
    def env_method(self, method_name: str, *method_args, indices = None, **method_kwargs) -> List[Any]:
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]
    
    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices) -> List[gym.Env]:
        indices = self._get_indices(0)
        return [self.envs[i] for i in indices]