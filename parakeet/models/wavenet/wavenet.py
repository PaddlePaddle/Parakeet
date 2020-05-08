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

from __future__ import division
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
    """Linearlly quantize a float Tensor in [-1, 1) to an interger Tensor in [0, n_bands).

    Args:
        values (Variable): dtype: flaot32 or float64. the floating point value.
        n_bands (int): the number of bands. The output integer Tensor's value is in the range [0, n_bans).

    Returns:
        Variable: the quantized tensor, dtype: int64.
    """
    quantized = F.cast((values + 1.0) / 2.0 * n_bands, "int64")
    return quantized


def dequantize(quantized, n_bands):
    """Linearlly dequantize an integer Tensor into a float Tensor in the range [-1, 1).

    Args:
        quantized (Variable): dtype: int64. The quantized value in the range [0, n_bands).
        n_bands (int): number of bands. The input integer Tensor's value is in the range [0, n_bans).

    Returns:
        Variable: the dequantized tensor, dtype float3232.
    """
    value = (F.cast(quantized, "float32") + 0.5) * (2.0 / n_bands) - 1.0
    return value


class ResidualBlock(dg.Layer):
    def __init__(self, residual_channels, condition_dim, filter_size,
                 dilation):
        """A Residual block in wavenet. It does not have parametric residual or skip connection. It consists of a Conv1DCell and an Conv1D(filter_size = 1) to integrate the condition.

        Args:
            residual_channels (int): the channels of the input, residual and skip.
            condition_dim (int): the channels of the condition.
            filter_size (int): filter size of the internal convolution cell.
            dilation (int): dilation of the internal convolution cell.
        """
        super(ResidualBlock, self).__init__()
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
        """Conv1D gated-tanh Block.

        Args:
            x (Variable): shape(B, C_res, T), the input. (B stands for batch_size, C_res stands for residual channels, T stands for time steps.) dtype float32.
            condition (Variable, optional): shape(B, C_cond, T), the condition, it has been upsampled in time steps, so it has the same time steps as the input does.(C_cond stands for the condition's channels). Defaults to None.

        Returns:
            (residual, skip_connection)
            residual (Variable): shape(B, C_res, T), the residual, which is used as the input to the next layer of ResidualBlock.
            skip_connection (Variable): shape(B, C_res, T), the skip connection. This output is accumulated with that of other ResidualBlocks. 
        """
        time_steps = x.shape[-1]
        h = x

        # dilated conv
        h = self.conv(h)
        if h.shape[-1] != time_steps:
            h = h[:, :, :time_steps]

        # condition
        if condition is not None:
            h += self.condition_proj(condition)

        # gated tanh
        content, gate = F.split(h, 2, dim=1)
        z = F.sigmoid(gate) * F.tanh(content)

        # projection
        residual = F.scale(z + x, math.sqrt(.5))
        skip_connection = z
        return residual, skip_connection

    def start_sequence(self):
        """Prepare the ResidualBlock to generate a new sequence. This method should be called before starting calling `add_input` multiple times.
        """
        self.conv.start_sequence()

    def add_input(self, x, condition=None):
        """Add a step input. This method works similarily with `forward` but in a `step-in-step-out` fashion.

        Args:
            x (Variable): shape(B, C_res, T=1), input for a step, dtype float32.
            condition (Variable, optional): shape(B, C_cond, T=1). condition for a step, dtype float32. Defaults to None.

        Returns:
            (residual, skip_connection)
            residual (Variable): shape(B, C_res, T=1), the residual for a step, which is used as the input to the next layer of ResidualBlock.
            skip_connection (Variable): shape(B, C_res, T=1), the skip connection for a step. This output is accumulated with that of other ResidualBlocks. 
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
        """The residual network in wavenet. It consists of `n_layer` stacks, each of which consists of `n_loop` ResidualBlocks.

        Args:
            n_loop (int): number of ResidualBlocks in a stack.
            n_layer (int): number of stacks in the `ResidualNet`.
            residual_channels (int): channels of each `ResidualBlock`'s input.
            condition_dim (int): channels of the condition.
            filter_size (int): filter size of the internal Conv1DCell of each `ResidualBlock`.
        """
        super(ResidualNet, self).__init__()
        # double the dilation at each layer in a loop(n_loop layers)
        dilations = [2**i for i in range(n_loop)] * n_layer
        self.context_size = 1 + sum(dilations)
        self.residual_blocks = dg.LayerList([
            ResidualBlock(residual_channels, condition_dim, filter_size,
                          dilation) for dilation in dilations
        ])

    def forward(self, x, condition=None):
        """
        Args:
            x (Variable): shape(B, C_res, T), dtype float32, the input. (B stands for batch_size, C_res stands for residual channels, T stands for time steps.)
            condition (Variable, optional): shape(B, C_cond, T), dtype float32, the condition, it has been upsampled in time steps, so it has the same time steps as the input does.(C_cond stands for the condition's channels) Defaults to None.

        Returns:
            skip_connection (Variable): shape(B, C_res, T), dtype float32, the output.
        """
        for i, func in enumerate(self.residual_blocks):
            x, skip = func(x, condition)
            if i == 0:
                skip_connections = skip
            else:
                skip_connections = F.scale(skip_connections + skip,
                                           np.sqrt(0.5))
        return skip_connections

    def start_sequence(self):
        """Prepare the ResidualNet to generate a new sequence. This method should be called before starting calling `add_input` multiple times.
        """
        for block in self.residual_blocks:
            block.start_sequence()

    def add_input(self, x, condition=None):
        """Add a step input. This method works similarily with `forward` but in a `step-in-step-out` fashion.

        Args:
            x (Variable): shape(B, C_res, T=1), dtype float32, input for a step.
            condition (Variable, optional): shape(B, C_cond, T=1), dtype float32, condition for a step. Defaults to None.

        Returns:
            skip_connection (Variable): shape(B, C_res, T=1), dtype float32, the output for a step.
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
        """Wavenet that transform upsampled mel spectrogram into waveform.

        Args:
            n_loop (int): n_loop for the internal ResidualNet.
            n_layer (int): n_loop for the internal ResidualNet.
            residual_channels (int): the channel of the input.
            output_dim (int): the channel of the output distribution. 
            condition_dim (int): the channel of the condition.
            filter_size (int): the filter size of the internal ResidualNet.
            loss_type (str): loss type of the wavenet. Possible values are 'softmax' and 'mog'. If `loss_type` is 'softmax', the output is the logits of the catrgotical(multinomial) distribution, `output_dim` means the number of classes of the categorical distribution. If `loss_type` is mog(mixture of gaussians), the output is the parameters of a mixture of gaussians, which consists of weight(in the form of logit) of each gaussian distribution and its mean and log standard deviaton. So when `loss_type` is 'mog', `output_dim` should be perfectly divided by 3.
            log_scale_min (int): the minimum value of log standard deviation of the output gaussian distributions. Note that this value is only used for computing loss if `loss_type` is 'mog', values less than `log_scale_min` is clipped when computing loss.
        """
        super(WaveNet, self).__init__()
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
        """compute the output distribution (represented by its parameters).

        Args:
            x (Variable): shape(B, T), dtype float32, the input waveform.
            condition (Variable, optional): shape(B, C_cond, T), dtype float32, the upsampled condition. Defaults to None.

        Returns:
            Variable: shape(B, T, C_output), dtype float32, the parameter of the output distributions.
        """

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
        """Prepare the WaveNet to generate a new sequence. This method should be called before starting calling `add_input` multiple times.
        """
        self.resnet.start_sequence()

    def add_input(self, x, condition=None):
        """compute the output distribution (represented by its parameters) for a step. It works similarily with the `forward` method but in a `step-in-step-out` fashion.

        Args:
            x (Variable): shape(B, T=1), dtype float32, a step of the input waveform.
            condition (Variable, optional): shape(B, C_cond, T=1), dtype float32, a step of the upsampled condition. Defaults to None.

        Returns:
            Variable: shape(B, T=1, C_output), dtype float32, the parameter of the output distributions.
        """
        # Causal Conv
        if self.loss_type == "softmax":
            x = F.clip(x, min=-1., max=0.99999)
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
        """compute the loss where output distribution is a categorial distribution.

        Args:
            y (Variable): shape(B, T, C_output), dtype float32, the logits of the output distribution.
            t (Variable): shape(B, T), dtype float32, the target audio. Note that the target's corresponding time index is one step ahead of the output distribution. And output distribution whose input contains padding is neglected in loss computation.

        Returns:
            Variable: shape(1, ), dtype float32, the loss.
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
        """Sample from the output distribution where the output distribution is a categorical distriobution.

        Args:
            y (Variable): shape(B, T, C_output), the logits of the output distribution

        Returns:
            Variable: shape(B, T), waveform sampled from the output distribution.
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
        """compute the loss where output distribution is a mixture of Gaussians.

        Args:
            y (Variable): shape(B, T, C_output), dtype float32, the parameterd of the output distribution. It is the concatenation of 3 parts, the logits of every distribution, the mean of each distribution and the log standard deviation of each distribution. Each part's shape is (B, T, n_mixture), where `n_mixture` means the number of Gaussians in the mixture.
            t (Variable): shape(B, T), dtype float32, the target audio. Note that the target's corresponding time index is one step ahead of the output distribution. And output distribution whose input contains padding is neglected in loss computation.

        Returns:
            Variable: shape(1, ), dtype float32, the loss.
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
        pdf_x = 1.0 / math.sqrt(2.0 * math.pi) * inv_std * exponent

        pdf_x = p_mixture * pdf_x
        # pdf_x: [bs, len]
        pdf_x = F.reduce_sum(pdf_x, dim=-1)
        per_sample_loss = -F.log(pdf_x + 1e-9)

        loss = F.reduce_mean(per_sample_loss)
        return loss

    def sample_from_mog(self, y):
        """Sample from the output distribution where the output distribution is a mixture of Gaussians.
        Args:
            y (Variable): shape(B, T, C_output), dtype float32, the parameterd of the output distribution. It is the concatenation of 3 parts, the logits of every distribution, the mean of each distribution and the log standard deviation of each distribution. Each part's shape is (B, T, n_mixture), where `n_mixture` means the number of Gaussians in the mixture.

        Returns:
            Variable: shape(B, T), waveform sampled from the output distribution.
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
        """Sample from the output distribution.
        Args:
            y (Variable): shape(B, T, C_output), dtype float32, the parameterd of the output distribution.

        Returns:
            Variable: shape(B, T), waveform sampled from the output distribution.
        """
        if self.loss_type == "softmax":
            return self.sample_from_softmax(y)
        else:
            return self.sample_from_mog(y)

    def loss(self, y, t):
        """compute the loss where output distribution is a mixture of Gaussians.

        Args:
            y (Variable): shape(B, T, C_output), dtype float32, the parameterd of the output distribution.
            t (Variable): shape(B, T), dtype float32, the target audio. Note that the target's corresponding time index is one step ahead of the output distribution. And output distribution whose input contains padding is neglected in loss computation.

        Returns:
            Variable: shape(1, ), dtype float32, the loss.
        """
        if self.loss_type == "softmax":
            return self.compute_softmax_loss(y, t)
        else:
            return self.compute_mog_loss(y, t)
