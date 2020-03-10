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
        """Clarinet model.

        Args:
            encoder (UpsampleNet): an UpsampleNet to upsample mel spectrogram.
            teacher (WaveNet): a WaveNet, the teacher.
            student (ParallelWaveNet): a ParallelWaveNet model, the student.
            stft (STFT): a STFT model to perform differentiable stft transform.
            min_log_scale (float, optional): used only for computing loss, the minimal value of log standard deviation of the output distribution of both the teacher and the student . Defaults to -6.0.
            lmd (float, optional): weight for stft loss. Defaults to 4.0.
        """
        super(Clarinet, self).__init__()
        self.encoder = encoder
        self.teacher = teacher
        self.student = student
        self.stft = stft

        self.lmd = lmd
        self.min_log_scale = min_log_scale

    def forward(self, audio, mel, audio_start, clip_kl=True):
        """Compute loss of Clarinet model.

        Args:
            audio (Variable): shape(B, T_audio), dtype flaot32, ground truth waveform.
            mel (Variable): shape(B, F, T_mel), dtype flaot32, condition(mel spectrogram here).
            audio_start (Variable): shape(B, ), dtype int64, audio starts positions.
            clip_kl (bool, optional): whether to clip kl_loss by maximum=100. Defaults to True.

        Returns:
            Dict(str, Variable)
            loss (Variable): shape(1, ), dtype flaot32, total loss.
            kl (Variable): shape(1, ), dtype flaot32, kl divergence between the teacher's output distribution and student's output distribution.
            regularization (Variable): shape(1, ), dtype flaot32, a regularization term of the KL divergence.
            spectrogram_frame_loss (Variable): shape(1, ), dytpe: float, stft loss, the L1-distance of the magnitudes of the spectrograms of the ground truth waveform and synthesized waveform.
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
        """Synthesize waveform using the encoder and the student network.

        Args:
            mel (Variable): shape(B, F, T_mel), the condition(mel spectrogram here).

        Returns:
            Variable: shape(B, T_audio), the synthesized waveform. (T_audio = T_mel * upscale_factor, where upscale_factor is the `upscale_factor` of the encoder.)
        """
        condition = self.encoder(mel)
        samples_shape = (condition.shape[0], condition.shape[-1])
        z = F.gaussian_random(samples_shape)
        x, s_means, s_scales = self.student(z, condition)
        return x


class STFT(dg.Layer):
    def __init__(self, n_fft, hop_length, win_length, window="hanning"):
        """A module for computing differentiable stft transform. See `librosa.stft` for more details.

        Args:
            n_fft (int): number of samples in a frame.
            hop_length (int): number of samples shifted between adjacent frames.
            win_length (int): length of the window function.
            window (str, optional): name of window function, see `scipy.signal.get_window` for more details. Defaults to "hanning".
        """
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
        """Compute the stft transform.

        Args:
            x (Variable): shape(B, T), dtype flaot32, the input waveform.

        Returns:
            (real, imag)
            real (Variable): shape(B, C, 1, T), dtype flaot32, the real part of the spectrogram. (C = 1 + n_fft // 2)
            imag (Variable): shape(B, C, 1, T), dtype flaot32, the image part of the spectrogram. (C = 1 + n_fft // 2) 
        """
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
        """Compute the power spectrogram.

        Args:
            (real, imag)
            real (Variable): shape(B, C, 1, T), dtype flaot32, the real part of the spectrogram.
            imag (Variable): shape(B, C, 1, T), dtype flaot32, the image part of the spectrogram.

        Returns:
            Variable: shape(B, C, 1, T), dtype flaot32, the power spectrogram.
        """
        real, imag = self(x)
        power = real**2 + imag**2
        return power

    def magnitude(self, x):
        """Compute the magnitude spectrogram.

        Args:
            (real, imag)
            real (Variable): shape(B, C, 1, T), dtype flaot32, the real part of the spectrogram.
            imag (Variable): shape(B, C, 1, T), dtype flaot32, the image part of the spectrogram.

        Returns:
            Variable: shape(B, C, 1, T), dtype flaot32, the magnitude spectrogram. It is the square root of the power spectrogram.
        """
        power = self.power(x)
        magnitude = F.sqrt(power)
        return magnitude
