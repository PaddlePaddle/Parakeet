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

import paddle
from paddle import nn
from paddle.nn import functional as F
from scipy import signal
import numpy as np

__all__ = ["quantize", "dequantize", "STFT"]


def quantize(values, n_bands):
    """Linearlly quantize a float Tensor in [-1, 1) to an interger Tensor in 
    [0, n_bands).

    Parameters
    -----------
    values : Tensor [dtype: flaot32 or float64]
        The floating point value.
        
    n_bands : int
        The number of bands. The output integer Tensor's value is in the range 
        [0, n_bans).

    Returns
    ----------
    Tensor [dtype: int 64]
        The quantized tensor.
    """
    quantized = paddle.cast((values + 1.0) / 2.0 * n_bands, "int64")
    return quantized


def dequantize(quantized, n_bands, dtype=None):
    """Linearlly dequantize an integer Tensor into a float Tensor in the range 
    [-1, 1).

    Parameters
    -----------
    quantized : Tensor [dtype: int]
        The quantized value in the range [0, n_bands).
        
    n_bands : int
        Number of bands. The input integer Tensor's value is in the range 
        [0, n_bans).
        
    dtype : str, optional
        Data type of the output.
        
    Returns
    -----------
    Tensor
        The dequantized tensor, dtype is specified by `dtype`. If `dtype` is 
        not specified, the default float data type is used.
    """
    dtype = dtype or paddle.get_default_dtype()
    value = (paddle.cast(quantized, dtype) + 0.5) * (2.0 / n_bands) - 1.0
    return value


class STFT(nn.Layer):
    """A module for computing stft transformation in a differentiable way. 
    
    Parameters
    ------------
    n_fft : int
        Number of samples in a frame.
        
    hop_length : int
        Number of samples shifted between adjacent frames.
        
    win_length : int
        Length of the window.
        
    window : str, optional
        Name of window function, see `scipy.signal.get_window` for more 
        details. Defaults to "hanning".
        
    Notes
    -----------
    It behaves like ``librosa.core.stft``. See ``librosa.core.stft`` for more 
    details.
    
    Given a audio which ``T`` samples, it the STFT transformation outputs a 
    spectrum with (C, frames) and complex dtype, where ``C = 1 + n_fft / 2`` 
    and ``frames = 1 + T // hop_lenghth``.
    
    Ony ``center`` and ``reflect`` padding is supported now.
    
    """

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
                            (self.n_bin, 1, 1, self.n_fft))
        w_imag = np.reshape(window *
                            np.sin(-2 * np.pi * M / n_fft)[:self.n_bin],
                            (self.n_bin, 1, 1, self.n_fft))

        w = np.concatenate([w_real, w_imag], axis=0)
        self.weight = paddle.cast(
            paddle.to_tensor(w), paddle.get_default_dtype())

    def forward(self, x):
        """Compute the stft transform.

        Parameters
        ------------
        x : Tensor [shape=(B, T)]
            The input waveform.

        Returns
        ------------
        real : Tensor [shape=(B, C, 1, frames)] 
            The real part of the spectrogram.
            
        imag : Tensor [shape=(B, C, 1, frames)] 
            The image part of the spectrogram.
        """
        # x(batch_size, time_steps)
        # pad it first with reflect mode
        # TODO(chenfeiyu): report an issue on paddle.flip
        pad_start = paddle.reverse(x[:, 1:1 + self.n_fft // 2], axis=[1])
        pad_stop = paddle.reverse(x[:, -(1 + self.n_fft // 2):-1], axis=[1])
        x = paddle.concat([pad_start, x, pad_stop], axis=-1)

        # to BC1T, C=1
        x = paddle.unsqueeze(x, axis=[1, 2])
        out = F.conv2d(x, self.weight, stride=(1, self.hop_length))
        real, imag = paddle.chunk(out, 2, axis=1)  # BC1T
        return real, imag

    def power(self, x):
        """Compute the power spectrum.

        Parameters
        ------------
        x : Tensor [shape=(B, T)]
            The input waveform.

        Returns
        ------------
        Tensor [shape=(B, C, 1, T)] 
            The power spectrum.
        """
        real, imag = self(x)
        power = real**2 + imag**2
        return power

    def magnitude(self, x):
        """Compute the magnitude of the spectrum.

        Parameters
        ------------
        x : Tensor [shape=(B, T)]
            The input waveform.

        Returns
        ------------
        Tensor [shape=(B, C, 1, T)] 
            The magnitude of the spectrum.
        """
        power = self.power(x)
        magnitude = paddle.sqrt(power)
        return magnitude
