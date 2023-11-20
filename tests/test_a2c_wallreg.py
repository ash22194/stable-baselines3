import torch
from torch import nn
import torch.nn.utils.prune as prune
import os
from ipdb import set_trace
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cmaps
from scipy.linalg import solve_continuous_are
from typing import Callable

from stable_baselines3.systems.linearsystem import LinearSystem
from stable_baselines3 import A2CwReg
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import SaveEveryCallback
from stable_baselines3.common.learning_schedules import linear_schedule
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

def plot_weights(model):
    # Visualize weights
    weights = model.policy.get_weights()
    weights_list = []
    max_weight = -torch.inf
    for key, value in weights.items():
        if (('policy-net' in key) or ('shared-net' in key)):
            value = torch.abs(value.data.flatten())
            weights_list += [value]
            max_value = torch.max(value)
            if (max_value > max_weight):
                max_weight = max_value

    fig, ax1 = plt.subplots(1,1)
    for ii in range(len(weights_list)):
        weight = weights_list[ii]        
        im1 = ax1.scatter(ii*np.ones(weight.shape), np.arange(weight.shape[0]), c=weight, vmin=0, vmax=max_weight, cmap=cmaps['plasma'])
    fig.colorbar(im1)
    plt.show()

def initialize_model(env, net_arch, n_steps_to_update, learning_rate, reg_coef, reg_type, save_every_timestep, directory:str='tests/data/linearsystem', algorithm:str='A2C', model_save_prefix: str ='model'):

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
        policy_std = 0.1
        policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=net_arch, log_std_init=np.log(policy_std))
        # Train model without any regularization
        model = A2CwReg('MlpPolicy', env, gamma=sys['gamma_'], reg_coef=reg_coef, reg_type=reg_type, n_steps=n_steps_to_update, learning_rate=learning_rate, tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)

    callback = SaveEveryCallback(save_every_timestep=save_every_timestep, save_path=save_path, save_prefix=model_save_prefix)

    return model, callback, save_timestep

def test_policy(model, env, starts=10, discount=1.):
    if (type(starts) == int):
        num_trajectories = starts
    elif (type(starts)==np.ndarray):
        num_trajectories = starts.shape[1]
    else:
        assert False

    trajectories = np.zeros((env.X_DIMS, int(env.T/env.dt)+1, num_trajectories))
    inputs = np.zeros((env.U_DIMS, int(env.T/env.dt), num_trajectories))
    costs = np.zeros(num_trajectories)
    for t in range(num_trajectories):
        if (type(starts) == int):
            obs = env.reset()
        else:
            obs = env.reset(starts[:,t])
        trajectories[:,0,t] = deepcopy(obs)
        for i in range(int(env.T/env.dt)):
            inputs[:,i,t], _state = model.predict(trajectories[:,i,t], deterministic=True)
            obs, reward, done, info = env.step(inputs[:,i,t])
            costs[t] = costs[t]*discount + reward
            trajectories[:,i+1,t] = deepcopy(obs)
            if done:
                print('Start state :', trajectories[:,0,t], ', Final state :', obs, ', Cost :', costs[t])
                break
    
    return trajectories, inputs, costs

if __name__=='__main__':
    # Define a double integrator environment
    sys = {'A': np.array([[0, 1], [0, 0]]), 'B': np.array([[0], [1]]), 'Q': np.diag([5, 0.25]), 'R': np.array([[0.01]]),\
        'goal': np.zeros((2,1)), 'u0': np.zeros((1,1)), 'T': 4, 'dt': 1e-3, 'gamma_': 0.997, 'X_DIMS': 2, 'U_DIMS': 1,\
        'x_limits': np.array([[-1, 1], [-3, 3]]), 'u_limits': np.array([[-50, 50]])}
    add_quad_feat = False
    normalized_actions = True
    env = LinearSystem(sys, add_quad_feat=add_quad_feat, normalized_actions=normalized_actions)
    check_env(env)

    # Save directory info
    directory = 'tests/data/linearsystem'
    algorithm = 'A2C'
    net_arch=[dict(pi=[16, 16, 16], vf=[16, 16, 16])]
    n_steps_to_update = 500
    learning_rate = 0.0015
    total_timesteps = 2000000
    save_every_timestep = 200000

    # No regularization
    model, callback, save_timestep = initialize_model(
        env,
        net_arch,
        n_steps_to_update,
        learning_rate=linear_schedule(learning_rate),
        reg_coef=0.,
        reg_type=None,
        save_every_timestep=save_every_timestep,
        directory=directory,
        algorithm=algorithm
    )
    model_load = save_timestep > 0
    if ((total_timesteps-model.num_timesteps) > 0):
        model.learn(total_timesteps=total_timesteps-model.num_timesteps, callback=callback, reset_num_timesteps=(not model_load))

    # Lasso
    model_lasso, callback_lasso, save_timestep_lasso = initialize_model(
        env,
        net_arch,
        n_steps_to_update,
        learning_rate=linear_schedule(learning_rate),
        reg_coef=1e-2,
        reg_type='lasso',
        save_every_timestep=save_every_timestep,
        directory=directory,
        algorithm=algorithm,
        model_save_prefix='model_lasso'
    )
    model_lasso_load = save_timestep_lasso > 0
    if ((total_timesteps-model_lasso.num_timesteps) > 0):
        model_lasso.learn(total_timesteps=total_timesteps-model_lasso.num_timesteps, callback=callback_lasso, reset_num_timesteps=(not model_lasso_load))

    # Group-lasso
    model_glasso, callback_glasso, save_timestep_glasso = initialize_model(
        env,
        net_arch,
        n_steps_to_update,
        learning_rate=linear_schedule(learning_rate),
        reg_coef=1e-2,
        reg_type='group-lasso',
        save_every_timestep=save_every_timestep,
        directory=directory,
        algorithm=algorithm,
        model_save_prefix='model_glasso'
    )
    model_glasso_load = save_timestep_glasso > 0
    if ((total_timesteps-model_glasso.num_timesteps) > 0):
        model_glasso.learn(total_timesteps=total_timesteps-model_glasso.num_timesteps, callback=callback_glasso, reset_num_timesteps=(not model_glasso_load))

    # Scale-lasso
    model_slasso, callback_slasso, save_timestep_slasso = initialize_model(
        env,
        net_arch,
        n_steps_to_update,
        learning_rate=linear_schedule(learning_rate),
        reg_coef=1e-2,
        reg_type='scale-lasso',
        save_every_timestep=save_every_timestep,
        directory=directory,
        algorithm=algorithm,
        model_save_prefix='model_slasso'
    )
    model_slasso_load = save_timestep_slasso > 0
    if ((total_timesteps-model_slasso.num_timesteps) > 0):
        model_slasso.learn(total_timesteps=total_timesteps-model_slasso.num_timesteps, callback=callback_slasso, reset_num_timesteps=(not model_slasso_load))

    # Plot weights
    plot_weights(model)
    plot_weights(model_lasso)
    plot_weights(model_glasso)
    plot_weights(model_slasso)

    # Test policy performance
    num_starts = 10
    print('No regularization')
    trajectories, inputs, costs = test_policy(model, env, starts=num_starts, discount=sys['gamma_'])
    print('With lasso regularization')
    trajectories_lasso, inputs_lasso, costs_lasso = test_policy(model_lasso, env, starts=deepcopy(np.reshape(trajectories[:,0,:], (sys['X_DIMS'], num_starts))), discount=sys['gamma_'])
    print('With group lasso regularization')
    trajectories_glasso, inputs_glasso, costs_glasso = test_policy(model_glasso, env, starts=deepcopy(np.reshape(trajectories[:,0,:], (sys['X_DIMS'], num_starts))), discount=sys['gamma_'])
    print('With scale lasso regularization')
    trajectories_slasso, inputs_slasso, costs_slasso = test_policy(model_slasso, env, starts=deepcopy(np.reshape(trajectories[:,0,:], (sys['X_DIMS'], num_starts))), discount=sys['gamma_'])

    set_trace()