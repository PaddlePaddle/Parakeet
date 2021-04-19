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
from typing import Union, Sequence, List
from tqdm import trange
import numpy as np

import paddle
from paddle import nn
from paddle.nn import functional as F
import paddle.fluid.initializer as I
import paddle.fluid.layers.distributions as D

from parakeet.modules.conv import Conv1dCell
from parakeet.modules.audio import quantize, dequantize, STFT
from parakeet.utils import checkpoint, layer_tools

__all__ = ["WaveNet", "ConditionalWaveNet"]


def crop(x, audio_start, audio_length):
    """Crop the upsampled condition to match audio_length. 
    
    The upsampled condition has the same time steps as the whole audio does. 
    But since audios are sliced to 0.5 seconds randomly while conditions are 
    not, upsampled conditions should also be sliced to extactly match the time 
    steps of the audio slice.

    Parameters
    ----------
    x : Tensor [shape=(B, C, T)]
        The upsampled condition.
    audio_start : Tensor [shape=(B,), dtype:int]
        The index of the starting point of the audio clips.
    audio_length : int
        The length of the audio clip(number of samples it contaions).

    Returns
    -------
    Tensor [shape=(B, C, audio_length)]
        Cropped condition.
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


class UpsampleNet(nn.LayerList):
    """A network used to upsample mel spectrogram to match the time steps of 
    audio.
    
    It consists of several layers of Conv2DTranspose. Each Conv2DTranspose 
    layer upsamples the time dimension by its `stride` times. 
    
    Also, each Conv2DTranspose's filter_size at frequency dimension is 3.

    Parameters
    ----------
    upscale_factors : List[int], optional
        Time upsampling factors for each Conv2DTranspose Layer. 
        
        The ``UpsampleNet`` contains ``len(upscale_factor)`` Conv2DTranspose 
        Layers. Each upscale_factor is used as the ``stride`` for the 
        corresponding Conv2DTranspose. Defaults to [16, 16], this the default 
        upsampling factor is 256.
        
    Notes
    ------
    ``np.prod(upscale_factors)`` should equals the ``hop_length`` of the stft 
    transformation used to extract spectrogram features from audio. 
    
    For example, ``16 * 16 = 256``, then the spectrogram extracted with a stft 
    transformation whose ``hop_length`` equals 256 is suitable. 
        
    See Also
    ---------
    ``librosa.core.stft``
    """

    def __init__(self, upscale_factors=[16, 16]):
        super(UpsampleNet, self).__init__()
        self.upscale_factors = list(upscale_factors)
        self.upscale_factor = np.prod(upscale_factors)

        for factor in self.upscale_factors:
            self.append(
                nn.utils.weight_norm(
                    nn.Conv2DTranspose(
                        1,
                        1,
                        kernel_size=(3, 2 * factor),
                        stride=(1, factor),
                        padding=(1, factor // 2))))

    def forward(self, x):
        r"""Compute the upsampled condition.

        Parameters
        -----------
        x : Tensor [shape=(B, F, T)]
            The condition (mel spectrogram here). ``F`` means the frequency 
            bands, which is the feature size of the input. 
            
            In the internal Conv2DTransposes, the frequency dimension 
            is treated as ``height`` dimension instead of ``in_channels``.

        Returns:
            Tensor [shape=(B, F, T \* upscale_factor)]
                The upsampled condition.
        """
        x = paddle.unsqueeze(x, 1)
        for sublayer in self:
            x = F.leaky_relu(sublayer(x), 0.4)
        x = paddle.squeeze(x, 1)
        return x


class ResidualBlock(nn.Layer):
    """A Residual block used in wavenet. Conv1D-gated-tanh Block.
        
    It consists of a Conv1DCell and an Conv1D(kernel_size = 1) to integrate 
    information of the condition.
    
    Notes
    --------
    It does not have parametric residual or skip connection. 

    Parameters
    -----------
    residual_channels : int
        The feature size of the input. It is also the feature size of the 
        residual output and skip output.
        
    condition_dim : int
        The feature size of the condition.
        
    filter_size : int
        Kernel size of the internal convolution cells.
        
    dilation :int
        Dilation of the internal convolution cells.
    """

    def __init__(self,
                 residual_channels: int,
                 condition_dim: int,
                 filter_size: Union[int, Sequence[int]],
                 dilation: int):

        super(ResidualBlock, self).__init__()
        dilated_channels = 2 * residual_channels
        # following clarinet's implementation, we do not have parametric residual
        # & skip connection.

        _filter_size = filter_size[0] if isinstance(filter_size, (
            list, tuple)) else filter_size
        std = math.sqrt(1 / (_filter_size * residual_channels))
        conv = Conv1dCell(
            residual_channels,
            dilated_channels,
            filter_size,
            dilation=dilation,
            weight_attr=I.Normal(scale=std))
        self.conv = nn.utils.weight_norm(conv)

        std = math.sqrt(1 / condition_dim)
        condition_proj = Conv1dCell(
            condition_dim,
            dilated_channels, (1, ),
            weight_attr=I.Normal(scale=std))
        self.condition_proj = nn.utils.weight_norm(condition_proj)

        self.filter_size = filter_size
        self.dilation = dilation
        self.dilated_channels = dilated_channels
        self.residual_channels = residual_channels
        self.condition_dim = condition_dim

    def forward(self, x, condition=None):
        """Forward pass of the ResidualBlock.

        Parameters
        -----------
        x : Tensor [shape=(B, C, T)]
            The input tensor.
             
        condition : Tensor, optional [shape(B, C_cond, T)]
            The condition. 
            
            It has been upsampled in time steps, so it has the same time steps 
            as the input does.(C_cond stands for the condition's channels). 
            Defaults to None.

        Returns
        -----------
        residual : Tensor [shape=(B, C, T)]
            The residual, which is used as the input to the next ResidualBlock.
            
        skip_connection : Tensor [shape=(B, C, T)]
            Tthe skip connection. This output is accumulated with that of 
            other ResidualBlocks. 
    """
        h = x
        length = x.shape[-1]
        
        # dilated conv
        h = self.conv(h)

        # condition
        # NOTE: expanded condition may have a larger timesteps than x
        if condition is not None:
            h += self.condition_proj(condition)[:, :, :length]

        # gated tanh
        content, gate = paddle.split(h, 2, axis=1)
        z = F.sigmoid(gate) * paddle.tanh(content)

        # projection
        residual = paddle.scale(z + x, math.sqrt(.5))
        skip_connection = z
        return residual, skip_connection

    def start_sequence(self):
        """Prepare the ResidualBlock to generate a new sequence. 
        
        Warnings
        ---------
        This method should be called before calling ``add_input`` multiple times.
        """
        self.conv.start_sequence()
        self.condition_proj.start_sequence()

    def add_input(self, x, condition=None):
        """Take a step input and return a step output. 
        
        This method works similarily with ``forward`` but in a 
        ``step-in-step-out`` fashion.

        Parameters
        ----------
        x : Tensor [shape=(B, C)]
            Input for a step.
            
        condition : Tensor, optional [shape=(B, C_cond)]
            Condition for a step. Defaults to None.

        Returns
        ----------
        residual : Tensor [shape=(B, C)] 
            The residual for a step, which is used as the input to the next 
            layer of ResidualBlock.
            
        skip_connection : Tensor [shape=(B, C)]
            T he skip connection for a step. This output is accumulated with 
            that of other ResidualBlocks. 
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
    """The residual network in wavenet. 
    
    It consists of ``n_stack`` stacks, each of which consists of ``n_loop``
    ResidualBlocks.

    Parameters
    ----------
    n_stack : int
        Number of stacks in the ``ResidualNet``.
        
    n_loop : int
        Number of ResidualBlocks in a stack.
        
    residual_channels : int
        Input feature size of each ``ResidualBlock``'s input.
        
    condition_dim : int
        Feature size of the condition.
        
    filter_size : int
        Kernel size of the internal ``Conv1dCell`` of each ``ResidualBlock``.

    """

    def __init__(self,
                 n_stack: int,
                 n_loop: int,
                 residual_channels: int,
                 condition_dim: int,
                 filter_size: int):
        super(ResidualNet, self).__init__()
        # double the dilation at each layer in a stack
        dilations = [2**i for i in range(n_loop)] * n_stack
        self.context_size = 1 + sum(dilations)
        for dilation in dilations:
            self.append(
                ResidualBlock(residual_channels, condition_dim, filter_size,
                              dilation))

    def forward(self, x, condition=None):
        """Forward pass of ``ResidualNet``.
        
        Parameters
        ----------
        x : Tensor [shape=(B, C, T)]
            The input. 
            
        condition : Tensor, optional [shape=(B, C_cond, T)]
            The condition, it has been upsampled in time steps, so it has the 
            same time steps as the input does. Defaults to None.

        Returns
        --------
        Tensor [shape=(B, C, T)]
            The output.
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
        should be called before starting calling ``add_input`` multiple times.
        """
        for block in self:
            block.start_sequence()

    def add_input(self, x, condition=None):
        """Take a step input and return a step output. 
        
        This method works similarily with ``forward`` but in a 
        ``step-in-step-out`` fashion.

        Parameters
        ----------
        x : Tensor [shape=(B, C)]
            Input for a step.
            
        condition : Tensor, optional [shape=(B, C_cond)]
            Condition for a step. Defaults to None.

        Returns
        ----------            
        Tensor [shape=(B, C)]
            The skip connection for a step. This output is accumulated with 
            that of other ResidualBlocks. 
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
    """Wavenet that transform upsampled mel spectrogram into waveform.

    Parameters
    -----------
    n_stack : int
        ``n_stack`` for the internal ``ResidualNet``.
        
    n_loop : int
        ``n_loop`` for the internal ``ResidualNet``.
        
    residual_channels : int
        Feature size of the input.
        
    output_dim : int
        Feature size of the input.
        
    condition_dim : int
        Feature size of the condition (mel spectrogram bands).
        
    filter_size : int
        Kernel size of the internal ``ResidualNet``.
        
    loss_type : str, optional ["mog" or "softmax"]
        The output type and loss type of the model, by default "mog".
        
        If "softmax", the model input is first quantized audio and the model 
        outputs a discret categorical distribution.
        
        If "mog", the model input is audio in floating point format, and the 
        model outputs parameters for a mixture of gaussian distributions. 
        Namely, the weight, mean and log scale of each gaussian distribution. 
        Thus, the ``output_size`` should be a multiple of 3.
    
    log_scale_min : float, optional
        Minimum value of the log scale of gaussian distributions, by default 
        -9.0.
        
        This is only used for computing loss when ``loss_type`` is "mog", If 
        the predicted log scale is less than -9.0, it is clipped at -9.0.
    """

    def __init__(self, n_stack, n_loop, residual_channels, output_dim,
                 condition_dim, filter_size, loss_type, log_scale_min):

        super(WaveNet, self).__init__()
        if loss_type not in ["softmax", "mog"]:
            raise ValueError("loss_type {} is not supported".format(loss_type))
        if loss_type == "softmax":
            self.embed = nn.Embedding(output_dim, residual_channels)
        else:
            if (output_dim % 3 != 0):
                raise ValueError(
                    "with Mixture of Gaussians(mog) output, the output dim must be divisible by 3, but get {}".
                    format(output_dim))
            self.embed = nn.utils.weight_norm(
                nn.Linear(1, residual_channels), dim=1)

        self.resnet = ResidualNet(n_stack, n_loop, residual_channels,
                                  condition_dim, filter_size)
        self.context_size = self.resnet.context_size

        skip_channels = residual_channels  # assume the same channel
        self.proj1 = nn.utils.weight_norm(
            nn.Linear(skip_channels, skip_channels), dim=1)
        self.proj2 = nn.utils.weight_norm(
            nn.Linear(skip_channels, skip_channels), dim=1)
        # if loss_type is softmax, output_dim is n_vocab of waveform magnitude.
        # if loss_type is mog, output_dim is 3 * gaussian, (weight, mean and stddev)
        self.proj3 = nn.utils.weight_norm(
            nn.Linear(skip_channels, output_dim), dim=1)

        self.loss_type = loss_type
        self.output_dim = output_dim
        self.input_dim = 1
        self.skip_channels = skip_channels
        self.log_scale_min = log_scale_min

    def forward(self, x, condition=None):
        """Forward pass of ``WaveNet``.

        Parameters
        -----------
        x : Tensor [shape=(B, T)] 
            The input waveform.
        condition : Tensor, optional [shape=(B, C_cond, T)]
            the upsampled condition. Defaults to None.

        Returns
        -------
        Tensor: [shape=(B, T, C_output)]
            The parameters of the output distributions.
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
        be called before starting calling ``add_input`` multiple times.
        """
        self.resnet.start_sequence()

    def add_input(self, x, condition=None):
        """Compute the output distribution (represented by its parameters) for 
        a step. It works similarily with the ``forward`` method but in a 
        ``step-in-step-out`` fashion.

        Parameters
        -----------
        x : Tensor [shape=(B,)]
            A step of the input waveform.
            
        condition : Tensor, optional [shape=(B, C_cond)]
            A step of the upsampled condition. Defaults to None.

        Returns
        --------
        Tensor: [shape=(B, C_output)]
            A step of the parameters of the output distributions.
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
        """Compute the loss when output distributions are categorial 
        distributions.

        Parameters
        ----------
        y : Tensor [shape=(B, T, C_output)]
            The logits of the output distributions.
            
        t : Tensor [shape=(B, T)]
            The target audio. The audio is first quantized then used as the 
            target.
            
        Notes
        -------
        Output distributions whose input contains padding is neglected in 
        loss computation. So the first ``context_size`` steps does not 
        contribute to the loss.

        Returns
        --------
        Tensor: [shape=(1,)]
            The loss.
        """
        # context size is not taken into account
        y = y[:, self.context_size:, :]
        t = t[:, self.context_size:]
        t = paddle.clip(t, min=-1.0, max=0.99999)
        quantized = quantize(t, n_bands=self.output_dim)
        label = paddle.unsqueeze(quantized, -1)

        loss = F.softmax_with_cross_entropy(y, label)
        reduced_loss = paddle.mean(loss)
        return reduced_loss

    def sample_from_softmax(self, y):
        """Sample from the output distribution when the output distributions 
        are categorical distriobutions.

        Parameters
        ----------
        y : Tensor [shape=(B, T, C_output)]
            The logits of the output distributions.

        Returns
        --------
        Tensor [shape=(B, T)]
            Waveform sampled from the output distribution.
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
        """Compute the loss where output distributions is a mixture of 
        Gaussians distributions.

        Parameters
        -----------
        y : Tensor [shape=(B, T, C_output)]
            The parameterd of the output distribution. It is the concatenation 
            of 3 parts, the logits of every distribution, the mean of each 
            distribution and the log standard deviation of each distribution. 
            
            Each part's shape is (B, T, n_mixture), where ``n_mixture`` means 
            the number of Gaussians in the mixture.
            
        t : Tensor [shape=(B, T)]
            The target audio. 
            
        Notes
        -------
        Output distributions whose input contains padding is neglected in 
        loss computation. So the first ``context_size`` steps does not 
        contribute to the loss.

        Returns
        --------
        Tensor: [shape=(1,)]
            The loss.
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
        pdf_x = paddle.sum(pdf_x, -1)
        per_sample_loss = -paddle.log(pdf_x + 1e-9)

        loss = paddle.mean(per_sample_loss)
        return loss

    def sample_from_mog(self, y):
        """Sample from the output distribution when the output distribution 
        is a mixture of Gaussian distributions.
        
        Parameters
        ------------
        y : Tensor [shape=(B, T, C_output)]
            The parameterd of the output distribution. It is the concatenation 
            of 3 parts, the logits of every distribution, the mean of each 
            distribution and the log standard deviation of each distribution. 
            
            Each part's shape is (B, T, n_mixture), where ``n_mixture`` means 
            the number of Gaussians in the mixture.

        Returns
        --------
        Tensor: [shape=(B, T)]
            Waveform sampled from the output distribution.
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
        
        Parameters
        ----------
        y : Tensor [shape=(B, T, C_output)]
            The parameterd of the output distribution.

        Returns
        --------
        Tensor [shape=(B, T)]
            Waveform sampled from the output distribution.
        """
        if self.loss_type == "softmax":
            return self.sample_from_softmax(y)
        else:
            return self.sample_from_mog(y)

    def loss(self, y, t):
        """Compute the loss given the output distribution and the target.

        Parameters
        ----------
        y : Tensor [shape=(B, T, C_output)]
            The parameters of the output distribution.
            
        t : Tensor [shape=(B, T)]
            The target audio.

        Returns
        ---------
        Tensor: [shape=(1,)]    
            The loss.
        """
        if self.loss_type == "softmax":
            return self.compute_softmax_loss(y, t)
        else:
            return self.compute_mog_loss(y, t)


class ConditionalWaveNet(nn.Layer):
    r"""Conditional Wavenet. An implementation of 
    `WaveNet: A Generative Model for Raw Audio <http://arxiv.org/abs/1609.03499>`_.
    
    It contains an UpsampleNet as the encoder and a WaveNet as the decoder. 
    It is an autoregressive model that generate raw audio.

    Parameters
    ----------
    upsample_factors : List[int]
        The upsampling factors of the UpsampleNet.
        
    n_stack : int
        Number of convolution stacks in the WaveNet. 
        
    n_loop : int
        Number of convolution layers in a convolution stack.
        
        Convolution layers in a stack have exponentially growing dilations, 
        from 1 to .. math:: `k^{n_{loop} - 1}`, where k is the kernel size.
        
    residual_channels : int
        Feature size of each ResidualBlocks.
        
    output_dim : int
        Feature size of the output. See ``loss_type`` for details.
        
    n_mels : int
        The number of bands of mel spectrogram.
        
    filter_size : int, optional
        Convolution kernel size of each ResidualBlock, by default 2.
        
    loss_type : str, optional ["mog" or "softmax"]
        The output type and loss type of the model, by default "mog".
        
        If "softmax", the model input should be quantized audio and the model 
        outputs a discret distribution.
        
        If "mog", the model input is audio in floating point format, and the 
        model outputs parameters for a mixture of gaussian distributions. 
        Namely, the weight, mean and logscale of each gaussian distribution. 
        Thus, the ``output_size`` should be a multiple of 3.
        
    log_scale_min : float, optional
        Minimum value of the log scale of gaussian distributions, by default 
        -9.0.
        
        This is only used for computing loss when ``loss_type`` is "mog", If 
        the predicted log scale is less than -9.0, it is clipped at -9.0.
    """

    def __init__(self,
                 upsample_factors: List[int],
                 n_stack: int,
                 n_loop: int,
                 residual_channels: int,
                 output_dim: int,
                 n_mels: int,
                 filter_size: int=2,
                 loss_type: str="mog",
                 log_scale_min: float=-9.0):
        super(ConditionalWaveNet, self).__init__()
        self.encoder = UpsampleNet(upsample_factors)
        self.decoder = WaveNet(
            n_stack=n_stack,
            n_loop=n_loop,
            residual_channels=residual_channels,
            output_dim=output_dim,
            condition_dim=n_mels,
            filter_size=filter_size,
            loss_type=loss_type,
            log_scale_min=log_scale_min)

    def forward(self, audio, mel):
        """Compute the output distribution given the mel spectrogram and the input(for teacher force training).

        Parameters
        -----------
        audio : Tensor [shape=(B, T_audio)]
            ground truth waveform, used for teacher force training.
            
        mel : Tensor [shape(B, F, T_mel)]
            Mel spectrogram. Note that it is the spectrogram for the whole 
            utterance.
            
        audio_start : Tensor [shape=(B,), dtype: int]
            Audio slices' start positions for each utterance.

        Returns
        ----------
        Tensor [shape(B, T_audio - 1, C_output)]
            Parameters for the output distribution, where ``C_output`` is the 
            ``output_dim`` of the decoder.)
        """
        audio_length = audio.shape[1]  # audio clip's length
        condition = self.encoder(mel)
        

        # shifting 1 step
        audio = audio[:, :-1]
        condition = condition[:, :, 1:]

        y = self.decoder(audio, condition)
        return y

    def loss(self, y, t):
        """Compute loss with respect to the output distribution and the target 
        audio.

        Parameters
        -----------
        y : Tensor [shape=(B, T - 1, C_output)]
            Parameters of the output distribution.
            
        t : Tensor [shape(B, T)] 
            target waveform.

        Returns
        --------
        Tensor: [shape=(1,)]
            the loss.
        """
        t = t[:, 1:]
        loss = self.decoder.loss(y, t)
        return loss

    def sample(self, y):
        """Sample from the output distribution.

        Parameters
        -----------
        y : Tensor [shape=(B, T, C_output)]
            Parameters of the output distribution.

        Returns
        --------
        Tensor [shape=(B, T)] 
            Sampled waveform from the output distribution.
        """
        samples = self.decoder.sample(y)
        return samples

    @paddle.no_grad()
    def infer(self, mel):
        r"""Synthesize waveform from mel spectrogram.

        Parameters
        -----------
        mel : Tensor [shape=(B, F, T)] 
            The ondition (mel spectrogram here).

        Returns
        -----------
        Tensor [shape=(B, T \* upsacle_factor)]
            Synthesized waveform.
            
            ``upscale_factor`` is the ``upscale_factor`` of the encoder 
            ``UpsampleNet``.
        """
        condition = self.encoder(mel)
        batch_size, _, time_steps = condition.shape
        samples = []

        self.decoder.start_sequence()
        x_t = paddle.zeros((batch_size, ), dtype=mel.dtype)
        for i in trange(time_steps):
            c_t = condition[:, :, i]  # (B, C)
            y_t = self.decoder.add_input(x_t, c_t)  #(B, C)
            y_t = paddle.unsqueeze(y_t, 1)
            x_t = self.sample(y_t)  # (B, 1)
            x_t = paddle.squeeze(x_t, 1)  #(B,)
            samples.append(x_t)
        samples = paddle.stack(samples, -1)
        return samples

    @paddle.no_grad()
    def predict(self, mel):
        r"""Synthesize audio from mel spectrogram. 
        
        The output and input are numpy arrays without batch.

        Parameters
        ----------
        mel : np.ndarray [shape=(C, T)]
            Mel spectrogram of an utterance.

        Returns
        -------
        Tensor : np.ndarray [shape=(C, T \* upsample_factor)]
            The synthesized waveform of an utterance.
        """
        mel = paddle.to_tensor(mel)
        mel = paddle.unsqueeze(mel, 0)
        audio = self.infer(mel)
        audio = audio[0].numpy()
        return audio

    @classmethod
    def from_pretrained(cls, config, checkpoint_path):
        """Build a ConditionalWaveNet model from a pretrained model.

        Parameters
        ----------        
        config: yacs.config.CfgNode
            model configs
        
        checkpoint_path: Path or str
            the path of pretrained model checkpoint, without extension name
        
        Returns
        -------
        ConditionalWaveNet
            The model built from pretrained result.
        """
        model = cls(upsample_factors=config.model.upsample_factors,
                    n_stack=config.model.n_stack,
                    n_loop=config.model.n_loop,
                    residual_channels=config.model.residual_channels,
                    output_dim=config.model.output_dim,
                    n_mels=config.data.n_mels,
                    filter_size=config.model.filter_size,
                    loss_type=config.model.loss_type,
                    log_scale_min=config.model.log_scale_min)
        layer_tools.summary(model)
        checkpoint.load_parameters(model, checkpoint_path=checkpoint_path)
        return model
