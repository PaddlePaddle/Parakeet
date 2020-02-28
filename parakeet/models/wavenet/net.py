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

import itertools
import numpy as np
from scipy import signal
from tqdm import trange

import paddle.fluid.layers as F
import paddle.fluid.dygraph as dg
import paddle.fluid.initializer as I
import paddle.fluid.layers.distributions as D

from parakeet.modules.weight_norm import Conv2DTranspose
from parakeet.models.wavenet.wavenet import WaveNet


def crop(x, audio_start, audio_length):
    """Crop mel spectrogram.
    
    Args:
        x (Variable): shape(batch_size, channels, time_steps), the condition, upsampled mel spectrogram.
        audio_start (int): starting point.
        audio_length (int): length.
    
    Returns:
        out: cropped condition.
    """

    # crop audio
    slices = []  # for each example
    starts = audio_start.numpy()
    for i in range(x.shape[0]):
        start = starts[i]
        end = start + audio_length
        slice = F.slice(x[i], axes=[1], starts=[start], ends=[end])
        slices.append(slice)
    out = F.stack(slices)
    return out


class UpsampleNet(dg.Layer):
    """A upsampling net (bridge net) in clarinet to upsample spectrograms from frame level to sample level.
    It consists of several(2) layers of transposed_conv2d. in time and frequency.
    The time dim is dilated hop_length times. The frequency bands retains.
    """

    def __init__(self, upscale_factors=[16, 16]):
        super(UpsampleNet, self).__init__()
        self.upscale_factors = list(upscale_factors)
        self.upsample_convs = dg.LayerList()
        for i, factor in enumerate(upscale_factors):
            self.upsample_convs.append(
                Conv2DTranspose(
                    1,
                    1,
                    filter_size=(3, 2 * factor),
                    stride=(1, factor),
                    padding=(1, factor // 2)))

    @property
    def upscale_factor(self):
        return np.prod(self.upscale_factors)

    def forward(self, x):
        """upsample local condition to match time steps of input signals. i.e. upsample mel spectrogram to match time steps for waveform, for each layer of a wavenet.
        
        Arguments:
            x {Variable} -- shape(batch_size, frequency, time_steps), local condition
        
        Returns:
            Variable -- shape(batch_size, frequency, time_steps * np.prod(upscale_factors)), upsampled condition for each layer.
        """
        x = F.unsqueeze(x, axes=[1])
        for sublayer in self.upsample_convs:
            x = F.leaky_relu(sublayer(x), alpha=.4)
        x = F.squeeze(x, [1])
        return x


# AutoRegressive Model
class ConditionalWavenet(dg.Layer):
    def __init__(self, encoder: UpsampleNet, decoder: WaveNet):
        super(ConditionalWavenet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, audio, mel, audio_start):
        """forward
        
        Arguments:
            audio {Variable} -- shape(batch_size, time_steps), waveform of 0.5 seconds
            mel {Variable} -- shape(batch_size, frequency_bands, frames), mel spectrogram of the whole sentence
            audio_start {Variable} -- shape(batch_size, ), audio start positions
        
        Returns:
            Variable -- shape(batch_size, time_steps - 1, output_dim), output distribution parameters
        """

        audio_length = audio.shape[1]  # audio clip's length
        condition = self.encoder(mel)
        condition_slice = crop(condition, audio_start,
                               audio_length)  # crop audio

        # shifting 1 step
        audio = audio[:, :-1]
        condition_slice = condition_slice[:, :, 1:]

        y = self.decoder(audio, condition_slice)
        return y

    def loss(self, y, t):
        """compute loss
        
        Arguments:
            y {Variable} -- shape(batch_size, time_steps - 1, output_dim), output distribution parameters
            t {Variable} -- shape(batch_size, time_steps), target waveform
        
        Returns:
            Variable -- shape(1, ), reduced loss 
        """
        t = t[:, 1:]
        loss = self.decoder.loss(y, t)
        return loss

    def sample(self, y):
        """sample from output distribution
        
        Arguments:
            y {Variable} -- shape(batch_size, time_steps, output_dim), output distribution parameters
        
        Returns:
            Variable -- shape(batch_size, time_steps) samples
        """

        samples = self.decoder.sample(y)
        return samples

    @dg.no_grad
    def synthesis(self, mel):
        """synthesize waveform from mel spectrogram
        
        Arguments:
            mel {Variable} -- shape(batch_size, frequency_bands, frames), mel-spectrogram
        
        Returns:
            Variable -- shape(batch_size, time_steps), synthesized waveform.
        """

        condition = self.encoder(mel)
        batch_size, _, time_steps = condition.shape
        samples = []

        self.decoder.start_sequence()
        x_t = F.zeros((batch_size, 1), dtype="float32")
        for i in trange(time_steps):
            c_t = condition[:, :, i:i + 1]
            y_t = self.decoder.add_input(x_t, c_t)
            x_t = self.sample(y_t)
            samples.append(x_t)

        samples = F.concat(samples, axis=-1)
        return samples
