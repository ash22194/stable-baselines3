import torch
from torch import nn
import numpy as np
from typing import List
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

    def __init__(self, in_dim: int, layers: List[int], out_dim: int):
        super().__init__()

        network: List[nn.Module] = []
        network.append(nn.Flatten())
        last_layer_dim = in_dim
        for ll in range(len(layers)):
            network.append(nn.Linear(last_layer_dim, layers[ll]))
            network.append(nn.ReLU())

            last_layer_dim = deepcopy(layers[ll])
        network.append(nn.Linear(last_layer_dim, out_dim))

        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)

    def get_scaling_weights(self):
        weights = dict()
        for key, value in self.named_parameters():
            if (('weight' in key) or ('bias' in key)):
                weights[key] = value
        
        return weights

    def l1_reg_l2_loss(self, x, y, reg_coeff=1e-2):
        weights = self.get_scaling_weights()
        y_hat = self.forward(x)
        pred_loss = torch.mean((y - y_hat)**2)
        reg_loss = torch.mean(torch.stack([torch.mean(torch.abs(w)) for k,w in weights.items()]))

        loss = pred_loss + reg_coeff * reg_loss

        return loss

class FullyConnectedNetworkwScaling(nn.Module):

    def __init__(self, in_dim: int, layers: List[int], out_dim: int):
        super().__init__()

        network: List[nn.Module] = []
        network.append(nn.Flatten())
        last_layer_dim = in_dim
        for ll in range(len(layers)):
            network.append(nn.Linear(last_layer_dim, layers[ll]))
            network.append(nn.ReLU())
            network.append(ScaleLayer(layers[ll]))

            last_layer_dim = deepcopy(layers[ll])
        network.append(nn.Linear(last_layer_dim, out_dim))

        self.network = nn.Sequential(*network)

    def forward(self, x):
        return self.network(x)
    
    def get_scaling_weights(self):
        weights = dict()
        for key, value in self.named_parameters():
            if ('scale' in key):
                weights[key] = value
        
        return weights
    
    def l1_reg_l2_loss(self, x, y, reg_coeff=1e-2):
        weights = self.get_scaling_weights()
        y_hat = self.forward(x)
        pred_loss = torch.mean((y - y_hat)**2)
        reg_loss = torch.mean(torch.stack([torch.mean(torch.abs(w)) for k,w in weights.items()]))

        loss = pred_loss + reg_coeff * reg_loss

        return loss

def plot_scaling_weights(model):
    # Visualize weights
    weights = model.get_scaling_weights()
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

    # Test forward pass to check if scaling works - does unit scaling return the same result as no scaling?
    # create a linear network with and without scaling
    FCN = FullyConnectedNetwork(input_dim, layers, output_dim)
    FCNwscal = FullyConnectedNetworkwScaling(input_dim, layers, output_dim)

    # copy weights form scale to nominal model
    scale_state_dict = FCNwscal.state_dict()
    state_dict_init = dict()
    for key, value in FCN.state_dict().items():
        layer_num = int(key.split('.')[1])
        lookup_lyer_num = divmod(layer_num, 2)[0]*3 + 1
        if ('weight' in key):
            lookup_layer_key = 'network.' + str(lookup_lyer_num) + '.weight'
        elif ('bias' in key):
            lookup_layer_key = 'network.' + str(lookup_lyer_num) + '.bias'
        state_dict_init[key] = scale_state_dict[lookup_layer_key].clone()
    FCN.load_state_dict(state_dict_init)

    num_samples = 100
    x = torch.rand(num_samples, input_dim)
    outx = FCN(x)
    outx_wscaling = FCNwscal(x)

    print('out - out_w_scale :', torch.max(torch.abs(outx_wscaling - outx)))

    # Test training
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
    FCNwscal_optimizer = torch.optim.SGD(FCNwscal.parameters(), lr=learning_rate)
    
    for ee in range(epochs):
        loss = FCN.l1_reg_l2_loss(x, f_x, reg_coeff=0.)
        loss_wscal = FCNwscal.l1_reg_l2_loss(x, f_x, reg_coeff=0.)

        loss.backward()
        FCN_optimizer.step()
        FCN_optimizer.zero_grad()

        loss_wscal.backward()
        FCNwscal_optimizer.step()
        FCNwscal_optimizer.zero_grad()

        with torch.no_grad():
            loss = FCN.l1_reg_l2_loss(x_test, f_x_test, reg_coeff=0.)
            loss_wscal = FCNwscal.l1_reg_l2_loss(x_test, f_x_test, reg_coeff=0.)

        print('Epoch : ', ee, ', Loss : ', loss, ', Loss (w scaling) : ', loss_wscal)

    # Test training with L1 regularization
    # check scaling before optimizing
    plot_scaling_weights(FCN)
    plot_scaling_weights(FCNwscal)
    print('Training with L1 regularization')
    reg_coeff = 1e-2
    for ee in range(epochs):
        loss_wreg = FCN.l1_reg_l2_loss(x, f_x, reg_coeff=reg_coeff)
        loss_wreg_wscal = FCNwscal.l1_reg_l2_loss(x, f_x, reg_coeff=reg_coeff)

        loss_wreg.backward()
        FCN_optimizer.step()
        FCN_optimizer.zero_grad()

        loss_wreg_wscal.backward()
        FCNwscal_optimizer.step()
        FCNwscal_optimizer.zero_grad()

        with torch.no_grad():
            loss_wreg = FCN.l1_reg_l2_loss(x_test, f_x_test, reg_coeff=0.)
            loss_wreg_wscal = FCNwscal.l1_reg_l2_loss(x_test, f_x_test, reg_coeff=0.)

        print('Epoch : ', ee, ', Loss : ', loss_wreg, ', Loss (w scaling) : ', loss_wreg_wscal)

    # check scaling after optimzing
    plot_scaling_weights(FCN)
    plot_scaling_weights(FCNwscal)