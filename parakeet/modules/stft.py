import paddle
from paddle import nn
from paddle.nn import functional as F
from scipy import signal
import numpy as np 

class STFT(nn.Layer):
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
                            (self.n_bin, 1, 1, self.n_fft))
        w_imag = np.reshape(window *
                            np.sin(-2 * np.pi * M / n_fft)[:self.n_bin],
                            (self.n_bin, 1, 1, self.n_fft))

        w = np.concatenate([w_real, w_imag], axis=0)
        self.weight = paddle.cast(paddle.to_tensor(w), paddle.get_default_dtype())

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
        magnitude = paddle.sqrt(power)
        return magnitude
