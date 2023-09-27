"""
Script to test lasso, group-lass and scale-lasso regularization for regression to a well defined function
"""

import torch
from torch import nn
import numpy as np
from typing import List, OrderedDict
from copy import deepcopy
from ipdb import set_trace
from matplotlib import pyplot as plt
from matplotlib import colormaps as cmaps

class ScaleLayer(nn.Module):
    def __init__(self, in_features: int, init_value=1.0):
        super().__init__()
        self.scale = nn.parameter.Parameter(init_value * torch.ones(in_features, requires_grad=True))

    def forward(self, input):
        return input * self.scale

class FullyConnectedNetwork(nn.Module):
    def __init__(self, in_dim: int, layers: List[int], out_dim: int, reg_type=None):
        super().__init__()

        if (reg_type=='scale-lasso'):
            with_scaling = True
        else:
            with_scaling = False
        self.reg_type = reg_type

        network: List[nn.Module] = []
        network.append(nn.Flatten())
        last_layer_dim = in_dim
        for ll in range(len(layers)):
            network.append(nn.Linear(last_layer_dim, layers[ll]))
            network.append(nn.ReLU())
            if (with_scaling):
                network.append(ScaleLayer(layers[ll]))

            last_layer_dim = deepcopy(layers[ll])
        network.append(nn.Linear(last_layer_dim, out_dim))

        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)

    def get_weights(self):
        weights = OrderedDict()
        for key, value in self.named_parameters():
            if (('weight' in key) or ('bias' in key) or ('scale' in key)):
                weights[key] = value
        
        return weights

    def get_loss(self, x, y, reg_coeff=0.):
        y_hat = self.forward(x)
        pred_loss = torch.mean((y - y_hat)**2)

        weights = self.get_weights()
        reg_loss = 0
        if (self.reg_type=='lasso'):
            reg_loss = torch.mean(torch.stack([torch.mean(torch.abs(w)) for k,w in weights.items()]))
        elif (self.reg_type=='scale-lasso'):
            reg_loss = torch.mean(torch.stack([torch.mean(torch.abs(v)) for k,v in weights.items() if 'scale' in k]))
        elif (self.reg_type=='group-lasso'):
            weights_list = list(weights.values())
            layer_l2_loss = []
            for ll in range(0, len(weights_list)-2, 2):
                layer_l2_loss += [
                        torch.sqrt(
                            torch.sum(weights_list[ll]**2, dim=1)     # weights in
                            + weights_list[ll+1]**2                   # bias terms
                            + torch.sum(weights_list[ll+2]**2, dim=0) # weights out
                        ) / (weights_list[ll].shape[1] + 1 + weights_list[ll+2].shape[0])
                    ]
            reg_loss = torch.mean(torch.stack(layer_l2_loss))

        loss = pred_loss + reg_coeff * reg_loss
        return loss

def plot_weights(model):
    # Visualize weights
    weights = model.get_weights()
    weights_list = []
    max_weight = -torch.inf
    for key, value in weights.items():
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

if __name__=='__main__':

    input_dim = 3
    layers = [64, 64, 64]
    output_dim = 1
    reg_coeff = 1e-2

    FCN = FullyConnectedNetwork(input_dim, layers, output_dim)
    FCNwlasso = FullyConnectedNetwork(input_dim, layers, output_dim, reg_type='lasso')
    FCNwgrouplasso = FullyConnectedNetwork(input_dim, layers, output_dim, reg_type='group-lasso')
    FCNwscalelasso = FullyConnectedNetwork(input_dim, layers, output_dim, reg_type='scale-lasso')

    num_samples = 100
    x = torch.rand(num_samples, input_dim)
    # define a function to approximate
    f = lambda y: 10 * torch.square(torch.norm(y, dim=1, keepdim=True)) - 4 * torch.norm(y, dim=1, keepdim=True) + 3
    f_x = f(x)

    num_test_samples = 10
    x_test = torch.rand(num_test_samples, input_dim)
    f_x_test = f(x_test)

    batch_size = num_samples
    epochs = 1000
    learning_rate = 1e-3
    FCN_optimizer = torch.optim.SGD(FCN.parameters(), lr=learning_rate)
    FCNwlasso_optimizer = torch.optim.SGD(FCNwlasso.parameters(), lr=learning_rate)
    FCNwgrouplasso_optimizer = torch.optim.SGD(FCNwgrouplasso.parameters(), lr=learning_rate)
    FCNwscalelasso_optimizer = torch.optim.SGD(FCNwscalelasso.parameters(), lr=learning_rate)

    for ee in range(epochs):
        loss = FCN.get_loss(x, f_x)
        loss_wlasso = FCNwlasso.get_loss(x, f_x, reg_coeff=reg_coeff)
        loss_wgrouplasso = FCNwgrouplasso.get_loss(x, f_x, reg_coeff=reg_coeff)
        loss_wscalelasso = FCNwscalelasso.get_loss(x, f_x, reg_coeff=reg_coeff)

        loss.backward()
        FCN_optimizer.step()
        FCN_optimizer.zero_grad()

        loss_wlasso.backward()
        FCNwlasso_optimizer.step()
        FCNwlasso_optimizer.zero_grad()

        loss_wgrouplasso.backward()
        FCNwgrouplasso_optimizer.step()
        FCNwgrouplasso_optimizer.zero_grad()

        loss_wscalelasso.backward()
        FCNwscalelasso_optimizer.step()
        FCNwscalelasso_optimizer.zero_grad()

        with torch.no_grad():
            loss = FCN.get_loss(x_test, f_x_test)
            loss_wlasso = FCNwlasso.get_loss(x_test, f_x_test)
            loss_wgrouplasso = FCNwgrouplasso.get_loss(x_test, f_x_test)
            loss_wscalelasso = FCNwscalelasso.get_loss(x_test, f_x_test)

        print('Epoch : ', ee, ', Loss : ', loss, ', w lasso : ', loss_wlasso, ', w g-lasso : ', loss_wgrouplasso, ', w s-lasso : ', loss_wscalelasso)

    # Visualize weights
    plot_weights(FCN)
    plot_weights(FCNwlasso)
    plot_weights(FCNwgrouplasso)
    plot_weights(FCNwscalelasso)
