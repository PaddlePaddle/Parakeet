import math
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I

from typing import Sequence
from parakeet.modules import geometry as geo

import itertools
import numpy as np
import paddle.fluid.dygraph as dg
from paddle import fluid

__all__ = ["WaveFlow"]

def fold(x, n_group):
    """Fold audio or spectrogram's temporal dimension in to groups.

    Args:
        x (Tensor): shape(*, time_steps), the input tensor
        n_group (int): the size of a group.

    Returns:
        Tensor: shape(*, time_steps // n_group, group), folded tensor.
    """
    *spatial_shape, time_steps = x.shape
    new_shape = spatial_shape + [time_steps // n_group, n_group]
    return paddle.reshape(x, new_shape)

class UpsampleNet(nn.LayerList):
    """
    Layer to upsample mel spectrogram to the same temporal resolution with 
    the corresponding waveform. It consists of several conv2dtranspose layers
    which perform de convolution on mel and time dimension.
    """
    def __init__(self, upsample_factors: Sequence[int]):
        super(UpsampleNet, self).__init__()
        for factor in upsample_factors:
            std = math.sqrt(1 / (3 * 2 * factor))
            init = I.Uniform(-std, std)
            self.append(
                nn.utils.weight_norm(
                    nn.Conv2DTranspose(1, 1, (3, 2 * factor), 
                        padding=(1, factor // 2),
                        stride=(1, factor),
                        weight_attr=init,
                        bias_attr=init)))
            
        # upsample factors
        self.upsample_factor = np.prod(upsample_factors)
        self.upsample_factors = upsample_factors
    
    def forward(self, x, trim_conv_artifact=False):
        """
        Args:
            x (Tensor): shape(batch_size, input_channels, time_steps), the input 
                spectrogram.
            trim_conv_artifact (bool, optional): trim deconvolution artifact at 
                each layer. Defaults to False.

        Returns:
            Tensor: shape(batch_size, input_channels, time_steps * upsample_factors).
                If trim_conv_artifact is True, the output time steps is less 
                than time_steps * upsample_factors.
        """
        x = paddle.unsqueeze(x, 1)
        for layer in self:
            x = layer(x)
            if trim_conv_artifact:
                time_cutoff = layer._kernel_size[1] - layer._stride[1]
                x = x[:, :, :, -time_cutoff]
            x = F.leaky_relu(x, 0.4)
        x = paddle.squeeze(x, 1)
        return x


class ResidualBlock(nn.Layer):
    """
    ResidualBlock that merges infomation from the condition and outputs residual 
    and skip. It has a conv2d layer, which has causal padding in height dimension 
    and same paddign in width dimension.
    """
    def __init__(self, channels, cond_channels, kernel_size, dilations):
        super(ResidualBlock, self).__init__()
        # input conv
        std = math.sqrt(1 / channels * np.prod(kernel_size))
        init = I.Uniform(-std, std)
        receptive_field = [1 + (k - 1) * d for (k, d) in zip(kernel_size, dilations)]
        rh, rw = receptive_field
        paddings = [rh - 1, 0, rw // 2, (rw - 1) // 2] # causal & same
        conv = nn.Conv2D(channels, 2 * channels, kernel_size, 
                         padding=paddings,
                         dilation=dilations, 
                         weight_attr=init, 
                         bias_attr=init)
        self.conv = nn.utils.weight_norm(conv)
        self.rh = rh
        self.rw = rw
        self.dilations = dilations
        
        # condition projection
        std = math.sqrt(1 / cond_channels)
        init = I.Uniform(-std, std)
        condition_proj = nn.Conv2D(cond_channels, 2 * channels, (1, 1),
                                   weight_attr=init, bias_attr=init)
        self.condition_proj = nn.utils.weight_norm(condition_proj)
        
        # parametric residual & skip connection
        std = math.sqrt(1 / channels)
        init = I.Uniform(-std, std)
        out_proj = nn.Conv2D(channels, 2 * channels, (1, 1),
                             weight_attr=init, bias_attr=init)
        self.out_proj = nn.utils.weight_norm(out_proj)
        
    def forward(self, x, condition):
        x = self.conv(x)
        x += self.condition_proj(condition)
        
        content, gate = paddle.chunk(x, 2, axis=1)
        x = paddle.tanh(content) * F.sigmoid(gate)
        
        x = self.out_proj(x)
        res, skip = paddle.chunk(x, 2, axis=1)
        return res, skip

    def start_sequence(self):
        if self.training:
            raise ValueError("Only use start sequence at evaluation mode.")
        self._conv_buffer = None

    def add_input(self, x_row, condition_row):
        if self._conv_buffer is None:
            self._init_buffer(x_row)
        self._update_buffer(x_row)

        rw = self.rw
        x_row = F.conv2d(
            self._conv_buffer,
            self.conv.weight,
            self.conv.bias,
            padding=[0, 0, rw // 2, (rw - 1) // 2],
            dilation=self.dilations)
        x_row += self.condition_proj(condition_row)

        content, gate = paddle.chunk(x_row, 2, axis=1)
        x_row = paddle.tanh(content) * F.sigmoid(gate)
        
        x_row = self.out_proj(x_row)
        res, skip = paddle.chunk(x_row, 2, axis=1)
        return res, skip

    def _init_buffer(self, input):
        batch_size, channels, _, width = input.shape
        self._conv_buffer = paddle.zeros(
            [batch_size, channels, self.rh, width], dtype=input.dtype)

    def _update_buffer(self, input):
        self._conv_buffer = paddle.concat(
            [self._conv_buffer[:, :, 1:, :], input], axis=2)


class ResidualNet(nn.LayerList):
    """
    A stack of several ResidualBlocks. It merges condition at each layer. All 
    skip outputs are collected.
    """
    def __init__(self, n_layer, residual_channels, condition_channels, kernel_size, dilations_h):
        if len(dilations_h) != n_layer:
            raise ValueError("number of dilations_h should equals num of layers")
        super(ResidualNet, self).__init__()
        for i in range(n_layer):
            dilation = (dilations_h[i], 2 ** i)
            layer = ResidualBlock(residual_channels, condition_channels, kernel_size, dilation)
            self.append(layer)
            
    def forward(self, x, condition):
        skip_connections = []
        for layer in self:
            x, skip = layer(x, condition)
            skip_connections.append(skip)
        out = paddle.sum(paddle.stack(skip_connections, 0), 0)
        return out

    def start_sequence(self):
        for layer in self:
            layer.start_sequence()
    
    def add_input(self, x_row, condition_row):
        # in reversed order
        skip_connections = []
        for layer in self:
            x_row, skip = layer.add_input(x_row, condition_row)
            skip_connections.append(skip)
        out = paddle.sum(paddle.stack(skip_connections, 0), 0)
        return out


class Flow(nn.Layer):
    """
    A Layer that merges the condition and predict scale and bias given a random
    variable X.
    """
    dilations_dict = {
            8: [1, 1, 1, 1, 1, 1, 1, 1],
            16: [1, 1, 1, 1, 1, 1, 1, 1],
            32: [1, 2, 4, 1, 2, 4, 1, 2],
            64: [1, 2, 4, 8, 16, 1, 2, 4],
            128: [1, 2, 4, 8, 16, 32, 64, 1]
    }
    
    def __init__(self, n_layers, channels, mel_bands, kernel_size, n_group):
        super(Flow, self).__init__()
        # input projection
        self.first_conv = nn.utils.weight_norm(
            nn.Conv2D(1, channels, (1, 1), 
                      weight_attr=I.Uniform(-1., 1.), 
                      bias_attr=I.Uniform(-1., 1.)))
        
        # residual net
        self.resnet = ResidualNet(n_layers, channels, mel_bands, kernel_size, 
                                  self.dilations_dict[n_group])
        
        # output projection
        self.last_conv = nn.utils.weight_norm(
            nn.Conv2D(channels, 2, (1, 1),
                      weight_attr=I.Constant(0.),
                      bias_attr=I.Constant(0.)))
        
        # specs
        self.n_group = n_group
    
    def _predict_parameters(self, x, condition):
        x = self.first_conv(x)
        x = self.resnet(x, condition)
        bijection_params = self.last_conv(x)
        logs, b = paddle.chunk(bijection_params, 2, axis=1)
        return logs, b

    def _transform(self, x, logs, b):
        z_0 = x[:, :, :1, :] # the first row, just copy it
        z_out = x[:, :, 1:, :] * paddle.exp(logs) + b            
        z_out = paddle.concat([z_0, z_out], axis=2)
        return z_out
    
    def forward(self, x, condition):
        # (B, C, H-1, W)
        logs, b = self._predict_parameters(
            x[:, :, :-1, :], condition[:, :, 1:, :]) 
        z = self._transform(x, logs, b)
        return z, (logs, b)

    def _predict_row_parameters(self, x_row, condition_row):
        x_row = self.first_conv(x_row)
        x_row = self.resnet.add_input(x_row, condition_row)
        bijection_params = self.last_conv(x_row)
        logs, b = paddle.chunk(bijection_params, 2, axis=1)
        return logs, b

    def _inverse_transform_row(self, z_row, logs, b):
        x_row = (z_row - b) / paddle.exp(logs)
        return x_row
    
    def _inverse_row(self, z_row, x_row, condition_row):
        logs, b = self._predict_row_parameters(x_row, condition_row)
        x_next_row = self._inverse_transform_row(z_row, logs, b)
        return x_next_row, (logs, b)

    def start_sequence(self):
        self.resnet.start_sequence()
    
    def inverse(self, z, condition):
        z_0 = z[:, :, :1, :]
        x = []
        logs_list = []
        b_list = []
        x.append(z_0)

        self.start_sequence()
        for i in range(1, self.n_group):
            x_row = x[-1] # actuallt i-1
            z_row = z[:, :, i:i+1, :]
            condition_row = condition[:, :, i:i+1, :]

            x_next_row, (logs, b) = self._inverse_row(z_row, x_row, condition_row)
            x.append(x_next_row)
            logs_list.append(logs)
            b_list.append(b)
        
        x = paddle.concat(x, 2)
        logs = paddle.concat(logs_list, 2)
        b = paddle.concat(b_list, 2)

        return x, (logs, b)

    
class WaveFlow(nn.LayerList):
    def __init__(self, n_flows, n_layers, n_group, channels, mel_bands, kernel_size):
        if n_group % 2 or n_flows % 2:
            raise ValueError("number of flows and number of group must be even "
                             "since a permutation along group among flows is used.")
        super(WaveFlow, self).__init__()
        for _ in range(n_flows):
            self.append(Flow(n_layers, channels, mel_bands, kernel_size, n_group))
        
        # permutations in h
        self.perms = self._create_perm(n_group, n_flows)

        # specs
        self.n_group = n_group
        self.n_flows = n_flows
    
    def _create_perm(self, n_group, n_flows):
        indices = list(range(n_group))
        half = n_group // 2
        perms = []
        for i in range(n_flows):
            if i < n_flows // 2:
                perms.append(indices[::-1])
            else:
                perm = list(reversed(indices[:half])) + list(reversed(indices[half:]))
                perms.append(perm)
        return perms
        
    def _trim(self, x, condition):
        assert condition.shape[-1] >= x.shape[-1]
        pruned_len = int(x.shape[-1] // self.n_group * self.n_group)
        
        if x.shape[-1] > pruned_len:
            x = x[:, :pruned_len]
        if condition.shape[-1] > pruned_len:
            condition = condition[:, :, :pruned_len]
        return x, condition
    
    def forward(self, x, condition):
        # x: (B, T)
        # condition: (B, C, T) upsampled condition
        x, condition = self._trim(x, condition)
        
        # to (B, C, h, T//h) layout
        x = paddle.unsqueeze(paddle.transpose(fold(x, self.n_group), [0, 2, 1]), 1)
        condition = paddle.transpose(fold(condition, self.n_group), [0, 1, 3, 2])
        
        # flows
        logs_list = []
        for i, layer in enumerate(self):
            x, (logs, b) = layer(x, condition)          
            logs_list.append(logs)
            # permute paddle has no shuffle dim
            x = geo.shuffle_dim(x, 2, perm=self.perms[i])
            condition = geo.shuffle_dim(condition, 2, perm=self.perms[i])

        z = paddle.squeeze(x, 1)
        return z, logs_list

    def start_sequence(self):
        for layer in self:
            layer.start_sequence()

    def inverse(self, z, condition):
        self.start_sequence()

        z, condition = self._trim(z, condition)
        # to (B, C, h, T//h) layout
        z = paddle.unsqueeze(paddle.transpose(fold(z, self.n_group), [0, 2, 1]), 1)
        condition = paddle.transpose(fold(condition, self.n_group), [0, 1, 3, 2])
        
        # reverse it flow by flow
        self.n_flows
        for i in reversed(range(self.n_flows)):
            z = geo.shuffle_dim(z, 2, perm=self.perms[i])
            condition = geo.shuffle_dim(condition, 2, perm=self.perms[i])
            z, (logs, b) = self[i].inverse(z, condition)
        x = paddle.squeeze(z, 1)
        return x


# TODO(chenfeiyu): WaveFlowLoss
