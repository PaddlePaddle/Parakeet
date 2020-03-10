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
    """Crop the upsampled condition to match audio_length. The upsampled condition has the same time steps as the whole audio does. But since audios are sliced to 0.5 seconds randomly while conditions are not, upsampled conditions should also be sliced to extaclt match the time steps of the audio slice.

    Args:
        x (Variable): shape(B, C, T), dtype float32, the upsample condition.
        audio_start (Variable): shape(B, ), dtype: int64, the index the starting point.
        audio_length (int): the length of the audio (number of samples it contaions).

    Returns:
        Variable: shape(B, C, audio_length), cropped condition.
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
    def __init__(self, upscale_factors=[16, 16]):
        """UpsamplingNet.
        It consists of several layers of Conv2DTranspose. Each Conv2DTranspose layer upsamples the time dimension by its `stride` times. And each Conv2DTranspose's filter_size at frequency dimension is 3.

        Args:
            upscale_factors (list[int], optional): time upsampling factors for each Conv2DTranspose Layer. The `UpsampleNet` contains len(upscale_factor) Conv2DTranspose Layers. Each upscale_factor is used as the `stride` for the corresponding Conv2DTranspose. Defaults to [16, 16].
        Note:
            np.prod(upscale_factors) should equals the `hop_length` of the stft transformation used to extract spectrogram features from audios. For example, 16 * 16 = 256, then the spectram extracted using a stft transformation whose `hop_length` is 256. See `librosa.stft` for more details.
        """
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
        """Compute the upsampled condition.

        Args:
            x (Variable): shape(B, F, T), dtype float32, the condition (mel spectrogram here.) (F means the frequency bands). In the internal Conv2DTransposes, the frequency dimension is treated as `height` dimension instead of `in_channels`.

        Returns:
            Variable: shape(B, F, T * upscale_factor), dtype float32, the upsampled condition.
        """
        x = F.unsqueeze(x, axes=[1])
        for sublayer in self.upsample_convs:
            x = F.leaky_relu(sublayer(x), alpha=.4)
        x = F.squeeze(x, [1])
        return x


# AutoRegressive Model
class ConditionalWavenet(dg.Layer):
    def __init__(self, encoder, decoder):
        """Conditional Wavenet, which contains an UpsampleNet as the encoder and a WaveNet as the decoder. It is an autoregressive model.

        Args:
            encoder (UpsampleNet): the UpsampleNet as the encoder.
            decoder (WaveNet): the WaveNet as the decoder.
        """
        super(ConditionalWavenet, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, audio, mel, audio_start):
        """Compute the output distribution given the mel spectrogram and the input(for teacher force training).

        Args:
            audio (Variable): shape(B, T_audio), dtype float32, ground truth waveform, used for teacher force training.
            mel ([Variable): shape(B, F, T_mel), dtype float32, mel spectrogram. Note that it is the spectrogram for the whole utterance.
            audio_start (Variable): shape(B, ), dtype: int, audio slices' start positions for each utterance.

        Returns:
            Variable: shape(B, T_audio - 1, C_putput), parameters for the output distribution.(C_output is the `output_dim` of the decoder.)
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
        """compute loss with respect to the output distribution and the targer audio.

        Args:
            y (Variable): shape(B, T - 1, C_output), dtype float32, parameters of the output distribution.
            t (Variable): shape(B, T), dtype float32, target waveform.

        Returns:
            Variable: shape(1, ), dtype float32, the loss.
        """
        t = t[:, 1:]
        loss = self.decoder.loss(y, t)
        return loss

    def sample(self, y):
        """Sample from the output distribution.

        Args:
            y (Variable): shape(B, T, C_output), dtype float32, parameters of the output distribution.

        Returns:
            Variable: shape(B, T), dtype float32, sampled waveform from the output distribution.
        """
        samples = self.decoder.sample(y)
        return samples

    @dg.no_grad
    def synthesis(self, mel):
        """Synthesize waveform from mel spectrogram.

        Args:
            mel (Variable): shape(B, F, T), condition(mel spectrogram here).

        Returns:
            Variable: shape(B, T * upsacle_factor), synthesized waveform.(`upscale_factor` is the `upscale_factor` of the encoder `UpsampleNet`)
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
