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
from tqdm import trange
import numpy as np

import paddle 
from paddle import nn
from paddle.nn import functional as F
import paddle.fluid.initializer as I
import paddle.fluid.layers.distributions as D

from parakeet.modules.conv import Conv1dCell

__all__ = ["ConditionalWavenet"]

def quantize(values, n_bands):
    """Linearlly quantize a float Tensor in [-1, 1) to an interger Tensor in [0, n_bands).

    Args:
        values (Tensor): dtype: flaot32 or float64. the floating point value.
        n_bands (int): the number of bands. The output integer Tensor's value is in the range [0, n_bans).

    Returns:
        Tensor: the quantized tensor, dtype: int64.
    """
    quantized = paddle.cast((values + 1.0) / 2.0 * n_bands, "int64")
    return quantized


def dequantize(quantized, n_bands, dtype=None):
    """Linearlly dequantize an integer Tensor into a float Tensor in the range [-1, 1).

    Args:
        quantized (Tensor): dtype: int64. The quantized value in the range [0, n_bands).
        n_bands (int): number of bands. The input integer Tensor's value is in the range [0, n_bans).

    Returns:
        Tensor: the dequantized tensor, dtype is specified by dtype.
    """
    dtype = dtype or paddle.get_default_dtype()
    value = (paddle.cast(quantized, dtype) + 0.5) * (2.0 / n_bands) - 1.0
    return value


def crop(x, audio_start, audio_length):
    """Crop the upsampled condition to match audio_length. The upsampled condition has the same time steps as the whole audio does. But since audios are sliced to 0.5 seconds randomly while conditions are not, upsampled conditions should also be sliced to extaclt match the time steps of the audio slice.

    Args:
        x (Tensor): shape(B, C, T), dtype float32, the upsample condition.
        audio_start (Tensor): shape(B,), dtype: int64, the index the starting point.
        audio_length (int): the length of the audio (number of samples it contaions).

    Returns:
        Tensor: shape(B, C, audio_length), cropped condition.
    """
    # crop audio
    slices = []  # for each example
    # paddle now supports Tensor of shape [1] in slice
    # starts = audio_start.numpy()
    for i in range(x.shape[0]):
        start = audio_start[i]
        end = start + audio_length
        slice = paddle.slice(x[i], axes=[1], starts=[start], ends=[end])
        slices.append(slice)
    out = paddle.stack(slices)
    return out


class ResidualBlock(nn.Layer):
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

        _filter_size = filter_size[0] if isinstance(filter_size, (list, tuple)) else filter_size
        std = math.sqrt(1 / (_filter_size * residual_channels))
        conv = Conv1dCell(residual_channels, 
                          dilated_channels, 
                          filter_size, 
                          dilation=dilation, 
                          weight_attr=I.Normal(scale=std))
        self.conv = nn.utils.weight_norm(conv)

        std = math.sqrt(1 / condition_dim)
        condition_proj = Conv1dCell(condition_dim, dilated_channels, (1,), 
                                   weight_attr=I.Normal(scale=std))
        self.condition_proj = nn.utils.weight_norm(condition_proj)

        self.filter_size = filter_size
        self.dilation = dilation
        self.dilated_channels = dilated_channels
        self.residual_channels = residual_channels
        self.condition_dim = condition_dim

    def forward(self, x, condition=None):
        """Conv1D gated-tanh Block.

        Args:
            x (Tensor): shape(B, C_res, T), the input. (B stands for batch_size, 
                C_res stands for residual channels, T stands for time steps.) 
                dtype float32.
            condition (Tensor, optional): shape(B, C_cond, T), the condition, 
                it has been upsampled in time steps, so it has the same time 
                steps as the input does.(C_cond stands for the condition's channels). 
                Defaults to None.

        Returns:
            (residual, skip_connection)
            residual (Tensor): shape(B, C_res, T), the residual, which is used 
                as the input to the next layer of ResidualBlock.
            skip_connection (Tensor): shape(B, C_res, T), the skip connection. 
                This output is accumulated with that of other ResidualBlocks. 
        """
        h = x

        # dilated conv
        h = self.conv(h)

        # condition
        if condition is not None:
            h += self.condition_proj(condition)

        # gated tanh
        content, gate = paddle.split(h, 2, axis=1)
        z = F.sigmoid(gate) * paddle.tanh(content)

        # projection
        residual = paddle.scale(z + x, math.sqrt(.5))
        skip_connection = z
        return residual, skip_connection

    def start_sequence(self):
        """
        Prepare the ResidualBlock to generate a new sequence. This method 
        should be called before starting calling `add_input` multiple times.
        """
        self.conv.start_sequence()
        self.condition_proj.start_sequence()

    def add_input(self, x, condition=None):
        """
        Add a step input. This method works similarily with `forward` but 
        in a `step-in-step-out` fashion.

        Args:
            x (Variable): shape(B, C_res), input for a step, dtype float32.
            condition (Variable, optional): shape(B, C_cond). condition for a 
                step, dtype float32. Defaults to None.

        Returns:
            (residual, skip_connection)
            residual (Variable): shape(B, C_res), the residual for a step, 
                which is used as the input to the next layer of ResidualBlock.
            skip_connection (Variable): shape(B, C_res), the skip connection 
                for a step. This output is accumulated with that of other 
                ResidualBlocks. 
        """
        h = x

        # dilated conv
        h = self.conv.add_input(h)

        # condition
        if condition is not None:
            h += self.condition_proj.add_input(condition)

        # gated tanh
        content, gate = paddle.split(h, 2, axis=1)
        z = F.sigmoid(gate) * paddle.tanh(content)

        # projection
        residual = paddle.scale(z + x, math.sqrt(0.5))
        skip_connection = z
        return residual, skip_connection


class ResidualNet(nn.LayerList):
    def __init__(self, n_loop, n_layer, residual_channels, condition_dim,
                 filter_size):
        """The residual network in wavenet. It consists of `n_layer` stacks, 
            each of which consists of `n_loop` ResidualBlocks.

        Args:
            n_loop (int): number of ResidualBlocks in a stack.
            n_layer (int): number of stacks in the `ResidualNet`.
            residual_channels (int): channels of each `ResidualBlock`'s input.
            condition_dim (int): channels of the condition.
            filter_size (int): filter size of the internal Conv1DCell of each 
                `ResidualBlock`.
        """
        super(ResidualNet, self).__init__()
        # double the dilation at each layer in a loop(n_loop layers)
        dilations = [2**i for i in range(n_loop)] * n_layer
        self.context_size = 1 + sum(dilations)
        for dilation in dilations:
            self.append(ResidualBlock(residual_channels, condition_dim, filter_size, dilation))

    def forward(self, x, condition=None):
        """
        Args:
            x (Tensor): shape(B, C_res, T), dtype float32, the input. 
                (B stands for batch_size, C_res stands for residual channels, 
                T stands for time steps.)
            condition (Tensor, optional): shape(B, C_cond, T), dtype float32, 
                the condition, it has been upsampled in time steps, so it has 
                the same time steps as the input does.(C_cond stands for the 
                condition's channels) Defaults to None.

        Returns:
            skip_connection (Tensor): shape(B, C_res, T), dtype float32, the output.
        """
        for i, func in enumerate(self):
            x, skip = func(x, condition)
            if i == 0:
                skip_connections = skip
            else:
                skip_connections = paddle.scale(skip_connections + skip,
                                        math.sqrt(0.5))
        return skip_connections

    def start_sequence(self):
        """Prepare the ResidualNet to generate a new sequence. This method 
            should be called before starting calling `add_input` multiple times.
        """
        for block in self:
            block.start_sequence()

    def add_input(self, x, condition=None):
        """Add a step input. This method works similarily with `forward` but 
            in a `step-in-step-out` fashion.

        Args:
            x (Tensor): shape(B, C_res), dtype float32, input for a step.
            condition (Tensor, optional): shape(B, C_cond), dtype float32, 
                condition for a step. Defaults to None.

        Returns:
            skip_connection (Tensor): shape(B, C_res), dtype float32, the 
                output for a step.
        """

        for i, func in enumerate(self):
            x, skip = func.add_input(x, condition)
            if i == 0:
                skip_connections = skip
            else:
                skip_connections = paddle.scale(skip_connections + skip,
                                        math.sqrt(0.5))
        return skip_connections


class WaveNet(nn.Layer):
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
            loss_type (str): loss type of the wavenet. Possible values are 
                'softmax' and 'mog'. 
                If `loss_type` is 'softmax', the output is the logits of the 
                catrgotical(multinomial) distribution, `output_dim` means the 
                number of classes of the categorical distribution. 
                If `loss_type` is mog(mixture of gaussians), the output is the 
                parameters of a mixture of gaussians, which consists of weight
                (in the form of logit) of each gaussian distribution and its 
                mean and log standard deviaton. So when `loss_type` is 'mog', 
                `output_dim` should be perfectly divided by 3.
            log_scale_min (int): the minimum value of log standard deviation 
                of the output gaussian distributions. Note that this value is 
                only used for computing loss if `loss_type` is 'mog', values 
                less than `log_scale_min` is clipped when computing loss.
        """
        super(WaveNet, self).__init__()
        if loss_type not in ["softmax", "mog"]:
            raise ValueError("loss_type {} is not supported".format(loss_type))
        if loss_type == "softmax":
            self.embed = nn.Embedding(output_dim, residual_channels)
        else:
            if (output_dim % 3 != 0):
                raise ValueError(
                    "with Mixture of Gaussians(mog) output, the output dim must be divisible by 3, but get {}".format(output_dim))
            self.embed = nn.utils.weight_norm(nn.Linear(1, residual_channels), dim=-1)

        self.resnet = ResidualNet(n_loop, n_layer, residual_channels,
                                  condition_dim, filter_size)
        self.context_size = self.resnet.context_size

        skip_channels = residual_channels  # assume the same channel
        self.proj1 = nn.utils.weight_norm(nn.Linear(skip_channels, skip_channels), dim=-1)
        self.proj2 = nn.utils.weight_norm(nn.Linear(skip_channels, skip_channels), dim=-1)
        # if loss_type is softmax, output_dim is n_vocab of waveform magnitude.
        # if loss_type is mog, output_dim is 3 * gaussian, (weight, mean and stddev)
        self.proj3 = nn.utils.weight_norm(nn.Linear(skip_channels, output_dim), dim=-1)

        self.loss_type = loss_type
        self.output_dim = output_dim
        self.input_dim = 1
        self.skip_channels = skip_channels
        self.log_scale_min = log_scale_min

    def forward(self, x, condition=None):
        """compute the output distribution (represented by its parameters).

        Args:
            x (Tensor): shape(B, T), dtype float32, the input waveform.
            condition (Tensor, optional): shape(B, C_cond, T), dtype float32, 
                the upsampled condition. Defaults to None.

        Returns:
            Tensor: shape(B, T, C_output), dtype float32, the parameter of 
            the output distributions.
        """

        # Causal Conv
        if self.loss_type == "softmax":
            x = paddle.clip(x, min=-1., max=0.99999)
            x = quantize(x, self.output_dim)
            x = self.embed(x)  # (B, T, C)
        else:
            x = paddle.unsqueeze(x, -1)  # (B, T, 1)
            x = self.embed(x)  # (B, T, C)
        x = paddle.transpose(x, perm=[0, 2, 1])  # (B, C, T)

        # Residual & Skip-conenection & linears
        z = self.resnet(x, condition)

        z = paddle.transpose(z, [0, 2, 1])
        z = F.relu(self.proj2(F.relu(self.proj1(z))))

        y = self.proj3(z)
        return y

    def start_sequence(self):
        """Prepare the WaveNet to generate a new sequence. This method should 
            be called before starting calling `add_input` multiple times.
        """
        self.resnet.start_sequence()

    def add_input(self, x, condition=None):
        """compute the output distribution (represented by its parameters) for 
            a step. It works similarily with the `forward` method but in a 
            `step-in-step-out` fashion.

        Args:
            x (Tensor): shape(B,), dtype float32, a step of the input waveform.
            condition (Tensor, optional): shape(B, C_cond, ), dtype float32, a 
                step of the upsampled condition. Defaults to None.

        Returns:
            Tensor: shape(B, C_output), dtype float32, the parameter of the 
                output distributions.
        """
        # Causal Conv
        if self.loss_type == "softmax":
            x = paddle.clip(x, min=-1., max=0.99999)
            x = quantize(x, self.output_dim)
            x = self.embed(x)  # (B, C)
        else:
            x = paddle.unsqueeze(x, -1)  # (B, 1)
            x = self.embed(x)  # (B, C)

        # Residual & Skip-conenection & linears
        z = self.resnet.add_input(x, condition)
        z = F.relu(self.proj2(F.relu(self.proj1(z))))  # (B, C)

        # Output
        y = self.proj3(z)
        return y

    def compute_softmax_loss(self, y, t):
        """compute the loss where output distribution is a categorial distribution.

        Args:
            y (Tensor): shape(B, T, C_output), dtype float32, the logits of the 
                output distribution.
            t (Tensor): shape(B, T), dtype float32, the target audio. Note that 
                the target's corresponding time index is one step ahead of the 
                output distribution. And output distribution whose input contains 
                padding is neglected in loss computation.

        Returns:
            Tensor: shape(1, ), dtype float32, the loss.
        """
        # context size is not taken into account
        y = y[:, self.context_size:, :]
        t = t[:, self.context_size:]
        t = paddle.clip(t, min=-1.0, max=0.99999)
        quantized = quantize(t, n_bands=self.output_dim)
        label = paddle.unsqueeze(quantized, -1)

        loss = F.softmax_with_cross_entropy(y, label)
        reduced_loss = paddle.reduce_mean(loss)
        return reduced_loss

    def sample_from_softmax(self, y):
        """Sample from the output distribution where the output distribution is 
            a categorical distriobution.

        Args:
            y (Tensor): shape(B, T, C_output), the logits of the output distribution.

        Returns:
            Tensor: shape(B, T), waveform sampled from the output distribution.
        """
        # dequantize
        batch_size, time_steps, output_dim, = y.shape
        y = paddle.reshape(y, (batch_size * time_steps, output_dim))
        prob = F.softmax(y)
        quantized = paddle.fluid.layers.sampling_id(prob)
        samples = dequantize(quantized, n_bands=self.output_dim)
        samples = paddle.reshape(samples, (batch_size, -1))
        return samples

    def compute_mog_loss(self, y, t):
        """compute the loss where output distribution is a mixture of Gaussians.

        Args:
            y (Tensor): shape(B, T, C_output), dtype float32, the parameterd of 
                the output distribution. It is the concatenation of 3 parts, 
                the logits of every distribution, the mean of each distribution 
                and the log standard deviation of each distribution. Each part's 
                shape is (B, T, n_mixture), where `n_mixture` means the number 
                of Gaussians in the mixture.
            t (Tensor): shape(B, T), dtype float32, the target audio. Note that 
                the target's corresponding time index is one step ahead of the 
                output distribution. And output distribution whose input contains 
                padding is neglected in loss computation.

        Returns:
            Tensor: shape(1, ), dtype float32, the loss.
        """
        n_mixture = self.output_dim // 3

        # context size is not taken in to account
        y = y[:, self.context_size:, :]
        t = t[:, self.context_size:]

        w, mu, log_std = paddle.split(y, 3, axis=2)
        # 100.0 is just a large float
        log_std = paddle.clip(log_std, min=self.log_scale_min, max=100.)
        inv_std = paddle.exp(-log_std)
        p_mixture = F.softmax(w, -1)

        t = paddle.unsqueeze(t, -1)
        if n_mixture > 1:
            # t = F.expand_as(t, log_std)
            t = paddle.expand(t, [-1, -1, n_mixture])

        x_std = inv_std * (t - mu)
        exponent = paddle.exp(-0.5 * x_std * x_std)
        pdf_x = 1.0 / math.sqrt(2.0 * math.pi) * inv_std * exponent

        pdf_x = p_mixture * pdf_x
        # pdf_x: [bs, len]
        pdf_x = paddle.reduce_sum(pdf_x, -1)
        per_sample_loss = -paddle.log(pdf_x + 1e-9)

        loss = paddle.reduce_mean(per_sample_loss)
        return loss

    def sample_from_mog(self, y):
        """Sample from the output distribution where the output distribution is 
            a mixture of Gaussians.
        Args:
            y (Tensor): shape(B, T, C_output), dtype float32, the parameterd of 
            the output distribution. It is the concatenation of 3 parts, the 
            logits of every distribution, the mean of each distribution and the 
            log standard deviation of each distribution. Each part's shape is 
            (B, T, n_mixture), where `n_mixture` means the number of Gaussians 
            in the mixture.

        Returns:
            Tensor: shape(B, T), waveform sampled from the output distribution.
        """
        batch_size, time_steps, output_dim = y.shape
        n_mixture = output_dim // 3

        w, mu, log_std = paddle.split(y, 3, -1)

        reshaped_w = paddle.reshape(w, (batch_size * time_steps, n_mixture))
        prob_ids = paddle.fluid.layers.sampling_id(F.softmax(reshaped_w))
        prob_ids = paddle.reshape(prob_ids, (batch_size, time_steps))
        prob_ids = prob_ids.numpy()

        # do it 
        index = np.array([[[b, t, prob_ids[b, t]] for t in range(time_steps)]
                          for b in range(batch_size)]).astype("int32")
        index_var = paddle.to_tensor(index)

        mu_ = paddle.gather_nd(mu, index_var)
        log_std_ = paddle.gather_nd(log_std, index_var)

        dist = D.Normal(mu_, paddle.exp(log_std_))
        samples = dist.sample(shape=[])
        samples = paddle.clip(samples, min=-1., max=1.)
        return samples

    def sample(self, y):
        """Sample from the output distribution.
        Args:
            y (Tensor): shape(B, T, C_output), dtype float32, the parameterd of 
                the output distribution.

        Returns:
            Tensor: shape(B, T), waveform sampled from the output distribution.
        """
        if self.loss_type == "softmax":
            return self.sample_from_softmax(y)
        else:
            return self.sample_from_mog(y)

    def loss(self, y, t):
        """compute the loss where output distribution is a mixture of Gaussians.

        Args:
            y (Tensor): shape(B, T, C_output), dtype float32, the parameterd of 
                the output distribution.
            t (Tensor): shape(B, T), dtype float32, the target audio. Note that 
                the target's corresponding time index is one step ahead of the 
                output distribution. And output distribution whose input contains 
                padding is neglected in loss computation.

        Returns:
            Tensor: shape(1, ), dtype float32, the loss.
        """
        if self.loss_type == "softmax":
            return self.compute_softmax_loss(y, t)
        else:
            return self.compute_mog_loss(y, t)


class UpsampleNet(nn.LayerList):
    def __init__(self, upscale_factors=[16, 16]):
        """UpsamplingNet.
        It consists of several layers of Conv2DTranspose. Each Conv2DTranspose 
            layer upsamples the time dimension by its `stride` times. And each 
            Conv2DTranspose's filter_size at frequency dimension is 3.

        Args:
            upscale_factors (list[int], optional): time upsampling factors for 
                each Conv2DTranspose Layer. The `UpsampleNet` contains 
                len(upscale_factor) Conv2DTranspose Layers. Each upscale_factor 
                is used as the `stride` for the corresponding Conv2DTranspose. 
                Defaults to [16, 16].
        Note:
            np.prod(upscale_factors) should equals the `hop_length` of the stft 
                transformation used to extract spectrogram features from audios. 
                For example, 16 * 16 = 256, then the spectram extracted using a 
                stft transformation whose `hop_length` is 256. See `librosa.stft` 
                for more details.
        """
        super(UpsampleNet, self).__init__()
        self.upscale_factors = list(upscale_factors)
        self.upscale_factor = 1
        for item in upscale_factors:
            self.upscale_factor *= item

        for factor in self.upscale_factors:
            self.append(
                nn.utils.weight_norm(
                    nn.ConvTranspose2d(1, 1, 
                        kernel_size=(3, 2 * factor), 
                        stride=(1, factor), 
                        padding=(1, factor // 2))))

    def forward(self, x):
        """Compute the upsampled condition.

        Args:
            x (Tensor): shape(B, F, T), dtype float32, the condition 
                (mel spectrogram here.) (F means the frequency bands). In the 
                internal Conv2DTransposes, the frequency dimension is treated 
                as `height` dimension instead of `in_channels`.

        Returns:
            Tensor: shape(B, F, T * upscale_factor), dtype float32, the 
                upsampled condition.
        """
        x = paddle.unsqueeze(x, 1)
        for sublayer in self:
            x = F.leaky_relu(sublayer(x), 0.4)
        x = paddle.squeeze(x, 1)
        return x


class ConditionalWavenet(nn.Layer):
    def __init__(self, encoder, decoder):
        """Conditional Wavenet, which contains an UpsampleNet as the encoder 
            and a WaveNet as the decoder. It is an autoregressive model.

        Args:
            encoder (UpsampleNet): the UpsampleNet as the encoder.
            decoder (WaveNet): the WaveNet as the decoder.
        """
        super(ConditionalWavenet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, audio, mel, audio_start):
        """Compute the output distribution given the mel spectrogram and the 
            input(for teacher force training).

        Args:
            audio (Tensor): shape(B, T_audio), dtype float32, ground truth 
                waveform, used for teacher force training.
            mel (Tensor): shape(B, F, T_mel), dtype float32, mel spectrogram. 
                Note that it is the spectrogram for the whole utterance.
            audio_start (Tensor): shape(B, ), dtype: int, audio slices' start 
                positions for each utterance.

        Returns:
            Tensor: shape(B, T_audio - 1, C_putput), parameters for the output 
                distribution.(C_output is the `output_dim` of the decoder.)
        """
        audio_length = audio.shape[1]  # audio clip's length
        condition = self.encoder(mel)
        condition_slice = crop(condition, audio_start, audio_length)

        # shifting 1 step
        audio = audio[:, :-1]
        condition_slice = condition_slice[:, :, 1:]

        y = self.decoder(audio, condition_slice)
        return y

    def loss(self, y, t):
        """compute loss with respect to the output distribution and the targer 
            audio.

        Args:
            y (Tensor): shape(B, T - 1, C_output), dtype float32, parameters of 
                the output distribution.
            t (Tensor): shape(B, T), dtype float32, target waveform.

        Returns:
            Tensor: shape(1, ), dtype float32, the loss.
        """
        t = t[:, 1:]
        loss = self.decoder.loss(y, t)
        return loss

    def sample(self, y):
        """Sample from the output distribution.

        Args:
            y (Tensor): shape(B, T, C_output), dtype float32, parameters of the 
                output distribution.

        Returns:
            Tensor: shape(B, T), dtype float32, sampled waveform from the output 
                distribution.
        """
        samples = self.decoder.sample(y)
        return samples

    @paddle.no_grad()
    def synthesis(self, mel):
        """Synthesize waveform from mel spectrogram.

        Args:
            mel (Tensor): shape(B, F, T), condition(mel spectrogram here).

        Returns:
            Tensor: shape(B, T * upsacle_factor), synthesized waveform.
                (`upscale_factor` is the `upscale_factor` of the encoder 
                `UpsampleNet`)
        """
        condition = self.encoder(mel)
        batch_size, _, time_steps = condition.shape
        samples = []

        self.decoder.start_sequence()
        x_t = paddle.zeros((batch_size, ), dtype=mel.dtype)
        for i in trange(time_steps):
            c_t = condition[:, :, i]
            y_t = self.decoder.add_input(x_t, c_t)
            y_t = paddle.unsqueeze(y_t, 1)
            x_t = self.sample(y_t)
            x_t = paddle.squeeze(x_t, 1)
            samples.append(x_t)

        samples = paddle.concat(samples, -1)
        return samples


# TODO WaveNetLoss