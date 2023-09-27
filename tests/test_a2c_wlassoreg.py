
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

from systems.linearsystem import LinearSystem
from stable_baselines3 import A2CwReg
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

def plot_abs_weights(model):
    # Visualize weights
    policy_weights = []
    policy_biases = []
    max_weight = -torch.inf
    for key, value in model.policy.get_weights().items():
        if ('policy-net' in key):
            value = torch.abs(value.data.flatten())
            if ('weight' in key):
                policy_weights += [value]
            elif ('bias' in key):
                policy_biases += [value]
            max_value = torch.max(value)
            if (max_value > max_weight):
                max_weight = max_value
    
    fig, (ax1, ax2) = plt.subplots(1,2)
    for ii in range(len(policy_weights)):
        weight = policy_weights[ii]
        bias = policy_biases[ii]

        im1 = ax1.scatter(ii*np.ones(weight.shape), np.arange(weight.shape[0]), c=weight, vmin=0, vmax=max_weight, cmap=cmaps['plasma'])
        im2 = ax2.scatter(ii*np.ones(bias.shape), np.arange(bias.shape[0]), c=bias, vmin=0, vmax=max_weight, cmap=cmaps['plasma'])

    fig.colorbar(im2)
    plt.show()

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

class SaveEveryCallback(BaseCallback):
    """
    A callback to periodically save the model that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, save_every_timestep, save_path, save_prefix='model', verbose=0):
        super(SaveEveryCallback, self).__init__(verbose)
        self.save_every_timestep = save_every_timestep

        assert os.path.isdir(save_path), 'Save directory does not exist!'
        self.save_path = save_path
        self.save_prefix = save_prefix

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        if (((self.model.num_timesteps+1) % self.save_every_timestep) == 0):
            self.model.save(os.path.join(self.save_path, self.save_prefix + '_' + str(self.model.num_timesteps+1)))

        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass

class ThresholdPruningMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self, threshold):
        super(ThresholdPruningMethod, self).__init__()
        self.threshold = threshold
    
    def compute_mask(self, t, default_mask):
        return torch.abs(t) > self.threshold

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
    save_path = os.path.join(directory, algorithm)
    log_path = os.path.join(save_path, 'tb_log')
    
    model_save_prefix = 'model'
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
    
    total_timesteps = 2000000 # cumulative steps across all environments
    save_every_timestep = 500000
    model_load = False
    if ((save_timestep <= total_timesteps) and (save_timestep > 0)):
        if (algorithm == 'A2C'):
            model = A2CwReg.load(os.path.join(save_path, model_save_prefix + '_' + str(save_timestep)))
        model.set_env(env)
        model_load = True
        total_timesteps -= model.num_timesteps
    else:
        n_steps_to_update = 500
        initial_learning_rate = 0.0015
        policy_std = 0.1
        policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[16, 16, 16], vf=[16, 16, 16])], log_std_init=np.log(policy_std))
        # Train model without any regularization
        model = A2CwReg('MlpPolicy', env, gamma=sys['gamma_'], reg_coef=0.0, n_steps=n_steps_to_update, learning_rate=linear_schedule(initial_learning_rate), tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
    
    callback = SaveEveryCallback(save_every_timestep=save_every_timestep, save_path=save_path)
    weights_before = model.policy.get_weights()
    weights_before = deepcopy(weights_before)
    model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=(not model_load))
    weights_after = model.policy.get_weights()
    weights_after = deepcopy(weights_after)
    plot_abs_weights(model)

    # # Load and intermediate model and check progress
    # set_trace()
    # model_trial = A2CwReg.load(os.path.join(save_path, model_save_prefix + '_' + str(save_every_timestep)))
    # set_trace()

    # Train model with regularization
    model_save_prefix = 'model_reg'
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

    total_timesteps = 2000000 # cumulative steps across all environments
    save_every_timestep = 500000
    n_steps_to_update = 500
    model_load = False
    if ((save_timestep <= total_timesteps) and (save_timestep > 0)):
        if (algorithm == 'A2C'):
            model_reg = A2CwReg.load(os.path.join(save_path, model_save_prefix + '_' + str(save_timestep)))
        model_reg.set_env(env)
        model_load = True
        total_timesteps -= model_reg.num_timesteps
    else:
        initial_learning_rate = 0.0015
        policy_std = 0.1
        policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=[dict(pi=[16, 16, 16], vf=[16, 16, 16])], log_std_init=np.log(policy_std))
        model_reg = A2CwReg('MlpPolicy', env, gamma=sys['gamma_'], reg_coef=0.01, reg_type='lasso', n_steps=n_steps_to_update, learning_rate=linear_schedule(initial_learning_rate), tensorboard_log=log_path, verbose=1, policy_kwargs=policy_kwargs)
    
    callback_reg = SaveEveryCallback(save_every_timestep=save_every_timestep, save_path=save_path, save_prefix='model_reg')
    model_reg_weights_before = model_reg.policy.get_weights()
    model_reg_weights_before = deepcopy(model_reg_weights_before)
    model_reg.learn(total_timesteps=total_timesteps, callback=callback_reg, reset_num_timesteps=(not model_load))
    model_reg_weights_after = model_reg.policy.get_weights()
    model_reg_weights_after = deepcopy(model_reg_weights_after)
    plot_abs_weights(model_reg)

    # Find a reasonable threshold to prune
    # First load the final saved model
    model_reg_pruned = A2CwReg.load(os.path.join(save_path, model_save_prefix + '_' + str(total_timesteps+model_reg.num_timesteps-n_steps_to_update)))
    model_reg_pruned.set_env(env)

    threshold = 0.15
    model_reg_pruned_policy_modules = model_reg_pruned.policy.mlp_extractor.shared_net + model_reg_pruned.policy.mlp_extractor.policy_net
    model_reg_pruned_policy_modules = [mm for mm in model_reg_pruned_policy_modules if (type(mm) == nn.modules.linear.Linear)]
    model_reg_pruned_policy_modules_weights = [(mm, 'weight') for mm in model_reg_pruned_policy_modules]
    model_reg_pruned_policy_modules_biases = [(mm, 'bias') for mm in model_reg_pruned_policy_modules]
    model_reg_pruned_policy_modules_tuple = model_reg_pruned_policy_modules_weights + model_reg_pruned_policy_modules_biases
    model_reg_pruned_policy_modules_tuple = tuple(model_reg_pruned_policy_modules_tuple)
    prune.global_unstructured(
        model_reg_pruned_policy_modules_tuple,
        pruning_method=ThresholdPruningMethod, threshold=threshold
    )
    # calculate number of parameters pruned
    weights_per_layer = [mm.weight_mask.numel() for mm in model_reg_pruned_policy_modules]
    biases_per_layer = [mm.bias_mask.numel() for mm in model_reg_pruned_policy_modules]
    nonzero_weights_per_layer = [torch.sum(mm.weight_mask) for mm in model_reg_pruned_policy_modules]
    nonzero_biases_per_layer = [torch.sum(mm.bias_mask) for mm in model_reg_pruned_policy_modules]
    set_trace()

    # Rollout trajectories and check how the policies behave
    num_trajectories = 4
    trajectories = np.zeros((sys['X_DIMS'], int(sys['T']/sys['dt'])+1, num_trajectories))
    print('Model without regularization')
    for t in range(num_trajectories):
        obs = env.reset()
        trajectories[:,0,t] = deepcopy(obs)
        for i in range(int(sys['T']/sys['dt'])):
            action, _state = model.predict(trajectories[:,i,t], deterministic=True)
            obs, reward, done, info = env.step(action)
            trajectories[:,i+1,t] = deepcopy(obs)
            if done:
                print('Start state :', trajectories[:,0,t], ', Final state :', obs)
                break

    print('Model with regularization')
    trajectories_reg = np.zeros((sys['X_DIMS'], int(sys['T']/sys['dt'])+1, num_trajectories))
    for t in range(num_trajectories):
        obs = env.reset(trajectories[:,0,t])
        trajectories_reg[:,0,t] = deepcopy(obs)
        for i in range(int(sys['T']/sys['dt'])):
            action, _state = model_reg.predict(trajectories_reg[:,i,t], deterministic=True)
            obs, reward, done, info = env.step(action)
            trajectories_reg[:,i+1,t] = deepcopy(obs)
            if done:
                print('Start state :', trajectories_reg[:,0,t], ', Final state :', obs)
                break
    
    print('Model with regularization (pruned)')
    trajectories_reg_pruned = np.zeros((sys['X_DIMS'], int(sys['T']/sys['dt'])+1, num_trajectories))
    for t in range(num_trajectories):
        obs = env.reset(trajectories[:,0,t])
        trajectories_reg_pruned[:,0,t] = deepcopy(obs)
        for i in range(int(sys['T']/sys['dt'])):
            action, _state = model_reg_pruned.predict(trajectories_reg_pruned[:,i,t], deterministic=True)
            obs, reward, done, info = env.step(action)
            trajectories_reg_pruned[:,i+1,t] = deepcopy(obs)
            if done:
                print('Start state :', trajectories_reg_pruned[:,0,t], ', Final state :', obs)
                break
    
    set_trace()