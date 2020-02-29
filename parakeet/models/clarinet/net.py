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
from parakeet.models.wavenet import crop, WaveNet, UpsampleNet
from parakeet.models.clarinet.parallel_wavenet import ParallelWaveNet
from parakeet.models.clarinet.utils import conv2d


# Gaussian IAF model
class Clarinet(dg.Layer):
    def __init__(self,
                 encoder,
                 teacher,
                 student,
                 stft,
                 min_log_scale=-6.0,
                 lmd=4.0):
        super(Clarinet, self).__init__()
        self.lmd = lmd
        self.encoder = encoder
        self.teacher = teacher
        self.student = student

        self.min_log_scale = min_log_scale
        self.stft = stft

    def forward(self, audio, mel, audio_start, clip_kl=True):
        """Compute loss for a distill model
        
        Arguments:
            audio {Variable} -- shape(batch_size, time_steps), target waveform.
            mel {Variable} -- shape(batch_size, condition_dim, time_steps // hop_length), original mel spectrogram, not upsampled yet.
            audio_starts {Variable} -- shape(batch_size, ), the index of the start sample.
            clip_kl (bool) -- whether to clip kl divergence if it is greater than 10.0.
        
        Returns:
            Variable -- shape(1,), loss
        """

        batch_size, audio_length = audio.shape  # audio clip's length

        z = F.gaussian_random(audio.shape)
        condition = self.encoder(mel)  # (B, C, T)
        condition_slice = crop(condition, audio_start, audio_length)

        x, s_means, s_scales = self.student(z, condition_slice)  # all [0: T]
        s_means = s_means[:, 1:]  # (B, T-1), time steps [1: T]
        s_scales = s_scales[:, 1:]  # (B, T-1), time steps [1: T]
        s_clipped_scales = F.clip(s_scales, self.min_log_scale, 100.)

        # teacher outputs single gaussian
        y = self.teacher(x[:, :-1], condition_slice[:, :, 1:])
        _, t_means, t_scales = F.split(y, 3, -1)  # time steps [1: T]
        t_means = F.squeeze(t_means, [-1])  # (B, T-1), time steps [1: T]
        t_scales = F.squeeze(t_scales, [-1])  # (B, T-1), time steps [1: T]
        t_clipped_scales = F.clip(t_scales, self.min_log_scale, 100.)

        s_distribution = D.Normal(s_means, F.exp(s_clipped_scales))
        t_distribution = D.Normal(t_means, F.exp(t_clipped_scales))

        # kl divergence loss, so we only need to sample once? no MC
        kl = s_distribution.kl_divergence(t_distribution)
        if clip_kl:
            kl = F.clip(kl, -100., 10.)
        # context size dropped
        kl = F.reduce_mean(kl[:, self.teacher.context_size:])
        # major diff here
        regularization = F.mse_loss(t_scales[:, self.teacher.context_size:],
                                    s_scales[:, self.teacher.context_size:])

        # introduce information from real target
        spectrogram_frame_loss = F.mse_loss(
            self.stft.magnitude(audio), self.stft.magnitude(x))
        loss = kl + self.lmd * regularization + spectrogram_frame_loss
        loss_dict = {
            "loss": loss,
            "kl_divergence": kl,
            "regularization": regularization,
            "stft_loss": spectrogram_frame_loss
        }
        return loss_dict

    @dg.no_grad
    def synthesis(self, mel):
        """Synthesize waveform conditioned on the mel spectrogram.
        
        Arguments:
            mel {Variable} -- shape(batch_size, frequqncy_bands, frames)
        
        Returns:
            Variable -- shape(batch_size, frames * upsample_factor)
        """
        condition = self.encoder(mel)
        samples_shape = (condition.shape[0], condition.shape[-1])
        z = F.gaussian_random(samples_shape)
        x, s_means, s_scales = self.student(z, condition)
        return x


class STFT(dg.Layer):
    def __init__(self, n_fft, hop_length, win_length, window="hanning"):
        super(STFT, self).__init__()
        self.hop_length = hop_length
        self.n_bin = 1 + n_fft // 2
        self.n_fft = n_fft

        # calculate window
        window = signal.get_window(window, win_length)
        if n_fft != win_length:
            pad = (n_fft - win_length) // 2
            window = np.pad(window, ((pad, pad), ), 'constant')

        # calculate weights
        r = np.arange(0, n_fft)
        M = np.expand_dims(r, -1) * np.expand_dims(r, 0)
        w_real = np.reshape(window *
                            np.cos(2 * np.pi * M / n_fft)[:self.n_bin],
                            (self.n_bin, 1, 1, self.n_fft)).astype("float32")
        w_imag = np.reshape(window *
                            np.sin(-2 * np.pi * M / n_fft)[:self.n_bin],
                            (self.n_bin, 1, 1, self.n_fft)).astype("float32")

        w = np.concatenate([w_real, w_imag], axis=0)
        self.weight = dg.to_variable(w)

    def forward(self, x):
        # x(batch_size, time_steps)
        # pad it first with reflect mode
        pad_start = F.reverse(x[:, 1:1 + self.n_fft // 2], axis=1)
        pad_stop = F.reverse(x[:, -(1 + self.n_fft // 2):-1], axis=1)
        x = F.concat([pad_start, x, pad_stop], axis=-1)

        # to BC1T, C=1
        x = F.unsqueeze(x, axes=[1, 2])
        out = conv2d(x, self.weight, stride=(1, self.hop_length))
        real, imag = F.split(out, 2, dim=1)  # BC1T
        return real, imag

    def power(self, x):
        real, imag = self(x)
        power = real**2 + imag**2
        return power

    def magnitude(self, x):
        power = self.power(x)
        magnitude = F.sqrt(power)
        return magnitude
