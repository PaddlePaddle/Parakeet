# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time
import itertools
import numpy as np

import paddle.fluid.layers as F
import paddle.fluid.dygraph as dg
import paddle.fluid.initializer as I
import paddle.fluid.layers.distributions as D

from parakeet.modules.weight_norm import Linear, Conv1D, Conv1DCell, Conv2DTranspose


# for wavenet with softmax loss
def quantize(values, n_bands):
    quantized = F.cast((values + 1.0) / 2.0 * n_bands, "int64")
    return quantized


def dequantize(quantized, n_bands):
    value = (F.cast(quantized, "float32") + 0.5) * (2.0 / n_bands) - 1.0
    return value


class ResidualBlock(dg.Layer):
    def __init__(self, residual_channels, condition_dim, filter_size,
                 dilation):
        super().__init__()
        dilated_channels = 2 * residual_channels
        # following clarinet's implementation, we do not have parametric residual
        # & skip connection.

        std = np.sqrt(1 / (filter_size * residual_channels))
        self.conv = Conv1DCell(
            residual_channels,
            dilated_channels,
            filter_size,
            dilation=dilation,
            causal=True,
            param_attr=I.Normal(scale=std))

        std = np.sqrt(1 / condition_dim)
        self.condition_proj = Conv1D(
            condition_dim, dilated_channels, 1, param_attr=I.Normal(scale=std))

        self.filter_size = filter_size
        self.dilation = dilation
        self.dilated_channels = dilated_channels
        self.residual_channels = residual_channels
        self.condition_dim = condition_dim

    def forward(self, x, condition=None):
        """Conv1D gated tanh Block
        
        Arguments:
            x {Variable} -- shape(batch_size, residual_channels, time_steps), the input.
        
        Keyword Arguments:
            condition {Variable} -- shape(batch_size, condition_dim, time_steps), upsampled local condition, it has the shape time steps as the input x. (default: {None})
        
        Returns:
            Variable -- shape(batch_size, residual_channels, time_steps), the output which is used as the input of the next layer.
            Variable -- shape(batch_size, residual_channels, time_steps), the output which is stacked alongside with other layers' as the output of wavenet.
        """
        time_steps = x.shape[-1]
        h = x

        # dilated conv
        h = self.conv(h)
        if h.shape[-1] != time_steps:
            h = h[:, :, :time_steps]

        # condition
        if condition:
            h += self.condition_proj(condition)

        # gated tanh
        content, gate = F.split(h, 2, dim=1)
        z = F.sigmoid(gate) * F.tanh(content)

        # projection
        residual = F.scale(z + x, math.sqrt(.5))
        skip_connection = z
        return residual, skip_connection

    def start_sequence(self):
        self.conv.start_sequence()

    def add_input(self, x, condition=None):
        """add a step input.
        
        Arguments:
            x {Variable} -- shape(batch_size, in_channels, time_steps=1), step input
        
        Keyword Arguments:
            condition {Variable} -- shape(batch_size, condition_dim, time_steps=1) (default: {None})
        
        Returns:
            Variable -- shape(batch_size, in_channels, time_steps=1), residual connection, which is the input for the next layer
            Variable -- shape(batch_size, in_channels, time_steps=1), skip connection
        """
        h = x

        # dilated conv
        h = self.conv.add_input(h)

        # condition
        if condition is not None:
            h += self.condition_proj(condition)

        # gated tanh
        content, gate = F.split(h, 2, dim=1)
        z = F.sigmoid(gate) * F.tanh(content)

        # projection
        residual = F.scale(z + x, np.sqrt(0.5))
        skip_connection = z
        return residual, skip_connection


class ResidualNet(dg.Layer):
    def __init__(self, n_loop, n_layer, residual_channels, condition_dim,
                 filter_size):
        super().__init__()
        # double the dilation at each layer in a loop(n_loop layers)
        dilations = [2**i for i in range(n_loop)] * n_layer
        self.context_size = 1 + sum(dilations)
        self.residual_blocks = dg.LayerList([
            ResidualBlock(residual_channels, condition_dim, filter_size,
                          dilation) for dilation in dilations
        ])

    def forward(self, x, condition=None):
        """n_layer layers of n_loop Residual Blocks.
        
        Arguments:
            x {Variable} -- shape(batch_size, residual_channels, time_steps), input of the residual net.
        
        Keyword Arguments:
            condition {Variable} -- shape(batch_size, condition_dim, time_steps), upsampled conditions, which has the same time steps as the input. (default: {None})
        
        Returns:
            Variable -- shape(batch_size, skip_channels, time_steps), output of the residual net.
        """

        #before_resnet = time.time()
        for i, func in enumerate(self.residual_blocks):
            x, skip = func(x, condition)
            if i == 0:
                skip_connections = skip
            else:
                skip_connections = F.scale(skip_connections + skip,
                                           np.sqrt(0.5))
        #print("resnet: ", time.time() - before_resnet)
        return skip_connections

    def start_sequence(self):
        for block in self.residual_blocks:
            block.start_sequence()

    def add_input(self, x, condition=None):
        """add step input and return step output.
        
        Arguments:
            x {Variable} -- shape(batch_size, residual_channels, time_steps=1), step input.
        
        Keyword Arguments:
            condition {Variable} -- shape(batch_size, condition_dim, time_steps=1), step condition (default: {None})
        
        Returns:
            Variable -- shape(batch_size, skip_channels, time_steps=1), step output, parameters of the output distribution.
        """

        for i, func in enumerate(self.residual_blocks):
            x, skip = func.add_input(x, condition)
            if i == 0:
                skip_connections = skip
            else:
                skip_connections = F.scale(skip_connections + skip,
                                           np.sqrt(0.5))
        return skip_connections


class WaveNet(dg.Layer):
    def __init__(self, n_loop, n_layer, residual_channels, output_dim,
                 condition_dim, filter_size, loss_type, log_scale_min):
        super().__init__()
        if loss_type not in ["softmax", "mog"]:
            raise ValueError("loss_type {} is not supported".format(loss_type))
        if loss_type == "softmax":
            self.embed = dg.Embedding((output_dim, residual_channels))
        else:
            assert output_dim % 3 == 0, "with MoG output, the output dim must be divided by 3"
            self.embed = Linear(1, residual_channels)

        self.resnet = ResidualNet(n_loop, n_layer, residual_channels,
                                  condition_dim, filter_size)
        self.context_size = self.resnet.context_size

        skip_channels = residual_channels  # assume the same channel
        self.proj1 = Linear(skip_channels, skip_channels)
        self.proj2 = Linear(skip_channels, skip_channels)
        # if loss_type is softmax, output_dim is n_vocab of waveform magnitude.
        # if loss_type is mog, output_dim is 3 * gaussian, (weight, mean and stddev)
        self.proj3 = Linear(skip_channels, output_dim)

        self.loss_type = loss_type
        self.output_dim = output_dim
        self.input_dim = 1
        self.skip_channels = skip_channels
        self.log_scale_min = log_scale_min

    def forward(self, x, condition=None):
        """(Possibly) Conditonal Wavenet.
        
        Arguments:
            x {Variable} -- shape(batch_size, time_steps), the input signal of wavenet. The waveform in 0.5 seconds.
        
        Keyword Arguments:
            conditions {Variable} -- shape(batch_size, condition_dim, 1, time_steps), the upsampled local condition. (default: {None})
        
        Returns:
            Variable -- shape(batch_size, time_steps, output_dim), output distributions at each time_steps. 
        """

        # CAUTION: rank-4 condition here
        # Causal Conv
        if self.loss_type == "softmax":
            x = F.clip(x, min=-1., max=0.99999)
            x = quantize(x, self.output_dim)
            x = self.embed(x)  # (B, T, C)
        else:
            x = F.unsqueeze(x, axes=[-1])  # (B, T, 1)
            x = self.embed(x)  # (B, T, C)
        x = F.transpose(x, perm=[0, 2, 1])  # (B, C, T)

        # Residual & Skip-conenection & linears
        z = self.resnet(x, condition)

        z = F.transpose(z, [0, 2, 1])
        z = F.relu(self.proj2(F.relu(self.proj1(z))))

        y = self.proj3(z)
        return y

    def start_sequence(self):
        self.resnet.start_sequence()

    def add_input(self, x, condition=None):
        """add step input
        
        Arguments:
            x {Variable} -- shape(batch_size, time_steps=1), step input.
        
        Keyword Arguments:
            condition {Variable} -- shape(batch_size, condition_dim , 1, time_steps=1) (default: {None})
        
        Returns:
            Variable -- ouput parameter for the distribution.
        """

        # Causal Conv
        if self.loss_type == "softmax":
            x = quantize(x, self.output_dim)
            x = self.embed(x)  # (B, T, C), T=1
        else:
            x = F.unsqueeze(x, axes=[-1])  # (B, T, 1), T=1
            x = self.embed(x)  # (B, T, C)
        x = F.transpose(x, perm=[0, 2, 1])

        # Residual & Skip-conenection & linears
        z = self.resnet.add_input(x, condition)
        z = F.transpose(z, [0, 2, 1])
        z = F.relu(self.proj2(F.relu(self.proj1(z))))  # (B, T, C)

        # Output
        y = self.proj3(z)
        return y

    def compute_softmax_loss(self, y, t):
        """compute loss, it is basically a language_model-like loss.
        
        Arguments:
            y {Variable} -- shape(batch_size, time_steps - 1, output_dim), output distribution of multinomial distribution.
            t {Variable} -- shape(batch_size, time_steps - 1), target waveform.
        
        Returns:
            Variable -- shape(1,), loss
        """

        # context size is not taken into account
        y = y[:, self.context_size:, :]
        t = t[:, self.context_size:]
        t = F.clip(t, min=-1.0, max=0.99999)
        quantized = quantize(t, n_bands=self.output_dim)
        label = F.unsqueeze(quantized, axes=[-1])

        loss = F.softmax_with_cross_entropy(y, label)
        reduced_loss = F.reduce_mean(loss)
        return reduced_loss

    def sample_from_softmax(self, y):
        """sample from output distribution.
        
        Arguments:
            y {Variable} -- shape(batch_size, time_steps - 1, output_dim), output distribution.
        
        Returns:
            Variable -- shape(batch_size, time_steps - 1), samples.
        """

        # dequantize
        batch_size, time_steps, output_dim, = y.shape
        y = F.reshape(y, (batch_size * time_steps, output_dim))
        prob = F.softmax(y)
        quantized = F.sampling_id(prob)
        samples = dequantize(quantized, n_bands=self.output_dim)
        samples = F.reshape(samples, (batch_size, -1))
        return samples

    def compute_mog_loss(self, y, t):
        """compute the loss with an mog output distribution.
        WARNING: this is not a legal probability, but a density. so it might be greater than 1.
        
        Arguments:
            y {Variable} -- shape(batch_size, time_steps, output_dim), output distribution's parameter. To represent a mixture of Gaussians. The output for each example at each time_step consists of 3 parts. The mean, the stddev, and a weight for that gaussian.
            t {Variable} -- shape(batch_size, time_steps), target waveform.

        Returns:
            Variable -- loss, note that it is computed with the pdf of the MoG distribution. 
        """

        n_mixture = self.output_dim // 3

        # context size is not taken in to account
        y = y[:, self.context_size:, :]
        t = t[:, self.context_size:]

        w, mu, log_std = F.split(y, 3, dim=2)
        # 100.0 is just a large float
        log_std = F.clip(log_std, min=self.log_scale_min, max=100.)
        inv_std = F.exp(-log_std)
        p_mixture = F.softmax(w, axis=-1)

        t = F.unsqueeze(t, axes=[-1])
        if n_mixture > 1:
            # t = F.expand_as(t, log_std)
            t = F.expand(t, [1, 1, n_mixture])

        x_std = inv_std * (t - mu)
        exponent = F.exp(-0.5 * x_std * x_std)
        pdf_x = 1.0 / np.sqrt(2.0 * np.pi) * inv_std * exponent
        pdf_x = p_mixture * pdf_x
        # pdf_x: [bs, len]
        pdf_x = F.reduce_sum(pdf_x, dim=-1)
        per_sample_loss = -F.log(pdf_x + 1e-9)

        loss = F.reduce_mean(per_sample_loss)
        return loss

    def sample_from_mog(self, y):
        """sample from output distribution.
        
        Arguments:
            y {Variable} -- shape(batch_size, time_steps - 1, output_dim), output distribution.
        
        Returns:
            Variable -- shape(batch_size, time_steps - 1), samples.
        """

        batch_size, time_steps, output_dim = y.shape
        n_mixture = output_dim // 3

        w, mu, log_std = F.split(y, 3, dim=-1)

        reshaped_w = F.reshape(w, (batch_size * time_steps, n_mixture))
        prob_ids = F.sampling_id(F.softmax(reshaped_w))
        prob_ids = F.reshape(prob_ids, (batch_size, time_steps))
        prob_ids = prob_ids.numpy()

        index = np.array([[[b, t, prob_ids[b, t]] for t in range(time_steps)]
                          for b in range(batch_size)]).astype("int32")
        index_var = dg.to_variable(index)

        mu_ = F.gather_nd(mu, index_var)
        log_std_ = F.gather_nd(log_std, index_var)

        dist = D.Normal(mu_, F.exp(log_std_))
        samples = dist.sample(shape=[])
        samples = F.clip(samples, min=-1., max=1.)
        return samples

    def sample(self, y):
        """sample from output distribution.
        
        Arguments:
            y {Variable} -- shape(batch_size, time_steps - 1, output_dim), output distribution.
        
        Returns:
            Variable -- shape(batch_size, time_steps - 1), samples.
        """

        if self.loss_type == "softmax":
            return self.sample_from_softmax(y)
        else:
            return self.sample_from_mog(y)

    def loss(self, y, t):
        """compute loss.
        
        Arguments:
            y {Variable} -- shape(batch_size, time_steps - 1, output_dim), output distribution of multinomial distribution.
            t {Variable} -- shape(batch_size, time_steps - 1), target waveform.
        
        Returns:
            Variable -- shape(1,), loss
        """

        if self.loss_type == "softmax":
            return self.compute_softmax_loss(y, t)
        else:
            return self.compute_mog_loss(y, t)
