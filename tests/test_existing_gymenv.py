import torch
from torch import nn
import os
import json
from copy import deepcopy
import gymnasium as gym
import numpy as np
from typing import Callable, Dict

from stable_baselines3 import A2CwReg
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.learning_schedules import linear_schedule
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

def initialize_model(env, algorithm_kwargs, policy_kwargs:Dict={}, directory:str='tests/data/linearsystem', algorithm:str='A2C', model_save_prefix: str ='model'):

    save_path = os.path.join(directory, algorithm)
    files = [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
    save_timestep = 0
    ff_latest = ''
    for ff in files:
        if model_save_prefix not in ff:
            continue 
        tt = ff.split('_')[-1]
        tt = int(tt.split('.')[0])
        if (tt > save_timestep):
            save_timestep = tt
            ff_latest = ff

    if (save_timestep > 0):
        if (algorithm == 'A2C'):
            model = A2CwReg.load(os.path.join(save_path, model_save_prefix + '_' + str(save_timestep)))
        model.set_env(env)
    else:
        log_path = os.path.join(save_path, 'tb_log')
        # Train model without any regularization
        model = A2CwReg('MlpPolicy', env, **algorithm_kwargs, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)

    return model

if __name__=='__main__':

    env_kwargs = dict()
    env = gym.make('Pendulum-v1', **env_kwargs)
    check_env(env)
    num_envs = 8
    env_vec = make_vec_env('Pendulum-v1', num_envs, env_kwargs=env_kwargs)

    # Save directory info
    directory = 'tests/data/pendulum'
    algorithm = 'A2C'
    initial_learning_rate = 7e-4
    learning_rate = linear_schedule(initial_learning_rate)

    algorithm_kwargs = {
        'max_grad_norm': 0.5,
        'n_steps': 8,
        'gamma': 0.9,
        'learning_rate': learning_rate,
        # 'vf_coef': 0.386,
        'reg_type': None,
        'reg_coef': 1e-2,
        'use_sde': False
    }
    policy_kwargs = dict(net_arch=[dict(pi=[32, 32], vf=[32, 32])], log_std_init=-2.3, ortho_init=True)

    total_timesteps = 1e6
    model_save_prefix = 'model'

    # Save (hyper)-parameters for the experiment to keep track
    with open(os.path.join(directory, algorithm, 'tb_log', 'parameters.txt'), 'w') as param_file:
        param = dict()
        param['algorithm'] = algorithm
        param.update(algorithm_kwargs)
        param['total_timesteps'] = total_timesteps
        param['model_save_prefix'] = model_save_prefix
        param.update(policy_kwargs)

        for key in param.keys():
            if (type(param[key]).__module__ == np.__name__):
                param[key] = param[key].tolist()
            elif (type(param[key]) == type):
                param[key] = param[key].__name__
            elif (type(param[key]) == type(linear_schedule)):
                param[key] = 'lin'+str(initial_learning_rate)

        param_file.write(json.dumps(param))

    # Initialize the model
    model = initialize_model(
        env_vec,
        algorithm_kwargs=algorithm_kwargs,
        policy_kwargs=policy_kwargs,
        directory=directory,
        algorithm=algorithm,
        model_save_prefix=model_save_prefix
    )

    # Train
    model_load = model.num_timesteps > 0
    if ((total_timesteps-model.num_timesteps) > 0):
        model.learn(total_timesteps=total_timesteps-model.num_timesteps, reset_num_timesteps=(not model_load))
    