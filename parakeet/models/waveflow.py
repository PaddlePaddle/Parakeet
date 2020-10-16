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
    def __init__(self, upsample_factors: Sequence[int]):
        super(UpsampleNet, self).__init__()
        for factor in upsample_factors:
            std = math.sqrt(1 / (3 * 2 * factor))
            init = I.Uniform(-std, std)
            self.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose2d(1, 1, (3, 2 * factor), 
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
    def __init__(self, channels, cond_channels, kernel_size, dilations):
        super(ResidualBlock, self).__init__()
        # input conv
        std = math.sqrt(1 / channels * np.prod(kernel_size))
        init = I.Uniform(-std, std)
        conv = nn.Conv2d(channels, 2 * channels, kernel_size, dilation=dilations, 
                         weight_attr=init, bias_attr=init)
        self.conv = nn.utils.weight_norm(conv)
        
        # condition projection
        std = math.sqrt(1 / cond_channels)
        init = I.Uniform(-std, std)
        condition_proj = nn.Conv2d(cond_channels, 2 * channels, (1, 1),
                                   weight_attr=init, bias_attr=init)
        self.condition_proj = nn.utils.weight_norm(condition_proj)
        
        # parametric residual & skip connection
        std = math.sqrt(1 / channels)
        init = I.Uniform(-std, std)
        out_proj = nn.Conv2d(channels, 2 * channels, (1, 1),
                                   weight_attr=init, bias_attr=init)
        self.out_proj = nn.utils.weight_norm(out_proj)
        
        # specs
        self.kernel_size = self.conv._kernel_size
        self.dilations = self.conv._dilation
        
    def forward(self, x, condition):
        receptive_field = tuple(
            [1 + (k -1) * d for (k, d) in zip(self.kernel_size, self.dilations)])
        rh, rw = receptive_field
        paddings = (rh - 1, 0, (rw - 1) // 2, (rw - 1) // 2)
        x = self.conv(F.pad2d(x, paddings))
        x += self.condition_proj(condition)
        
        content, gate = paddle.chunk(x, 2, axis=1)
        x = paddle.tanh(content) * F.sigmoid(gate)
        
        x = self.out_proj(x)
        res, skip = paddle.chunk(x, 2, axis=1)
        return res, skip
        
        
class ResidualNet(nn.LayerList):
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
    

class Flow(nn.Layer):
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
            nn.Conv2d(1, channels, (1, 1), 
                      weight_attr=I.Uniform(-1., 1.), 
                      bias_attr=I.Uniform(-1., 1.)))
        
        # residual net
        self.resnet = ResidualNet(n_layers, channels, mel_bands, kernel_size, 
                                  self.dilations_dict[n_group])
        
        # output projection
        self.last_conv = nn.utils.weight_norm(
            nn.Conv2d(channels, 2, (1, 1),
                      weight_attr=I.Constant(0.),
                      bias_attr=I.Constant(0.)))
    
    def forward(self, x, condition):
        return self.last_conv(self.resnet(self.first_conv(x), condition))


class WaveFlow(nn.LayerList):
    def __init__(self, n_flows, n_layers, n_group, channels, mel_bands, kernel_size):
        if n_group % 2 or n_flows % 2:
            raise ValueError("number of flows and number of group must be even "
                             "since a permutation along group among flows is used.")
        super(WaveFlow, self).__init__()
        for i in range(n_flows):
            self.append(Flow(n_layers, channels, mel_bands, kernel_size, n_group))
        
        # permutations in h
        indices = list(range(n_group))
        half = n_group // 2
        self.perms = []
        for i in range(n_flows):
            if i < n_flows // 2:
                self.perms.append(indices[::-1])
            else:
                perm = list(reversed(indices[:half])) + list(reversed(indices[half:]))
                self.perms.append(perm)
                
        self.n_group = n_group
        
    def trim(self, x, condition):
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
        x, condition = self.trim(x, condition)
        
        # transpose to (B, C, h, T //h) layout
        x = paddle.unsqueeze(paddle.transpose(fold(x, self.n_group), [0, 2, 1]), 1)
        condition = paddle.transpose(fold(condition, self.n_group), [0, 1, 3, 2])
        
        # flows
        logs_list = []
        for i, layer in enumerate(self):
            # shiting: z[i, j] depends only on x[<i, :]
            input = x[:, :, :-1, :]
            cond = condition[:, :, 1:, :]
            output = layer(input, cond)
            logs, b = paddle.chunk(output, 2, axis=1)
            logs_list.append(logs)

            x_0 = x[:, :, :1, :] # the first row, just  copy
            x_out = x[:, :, 1:, :] * paddle.exp(logs) + b            
            x = paddle.concat([x_0, x_out], axis=2)
            
            # permute paddle has no shuffle dim
            x = geo.shuffle_dim(x, 2, perm=self.perms[i])
            condition = geo.shuffle_dim(condition, 2, perm=self.perms[i])
        
        z = paddle.squeeze(x, 1)
        return z, logs_list


# TODO(chenfeiyu): WaveFlowLoss