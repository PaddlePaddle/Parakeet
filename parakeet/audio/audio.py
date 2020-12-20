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

import librosa
import soundfile as sf
import numpy as np

__all__ = ["AudioProcessor"]


class AudioProcessor(object):
    def __init__(self,
                 sample_rate: int,
                 n_fft: int,
                 win_length: int,
                 hop_length: int,
                 n_mels: int=80,
                 f_min: int=0,
                 f_max: int=None,
                 window="hann",
                 center=True,
                 pad_mode="reflect"):
        # read & write
        self.sample_rate = sample_rate

        # stft
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        # mel
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        self.mel_filter = self._create_mel_filter()
        self.inv_mel_filter = np.linalg.pinv(self.mel_filter)

    def _create_mel_filter(self):
        mel_filter = librosa.filters.mel(self.sample_rate,
                                         self.n_fft,
                                         n_mels=self.n_mels,
                                         fmin=self.f_min,
                                         fmax=self.f_max)
        return mel_filter

    def read_wav(self, filename):
        # resampling may occur
        wav, _ = librosa.load(filename, sr=self.sample_rate)
        return wav

    def write_wav(self, path, wav):
        sf.write(path, wav, samplerate=self.sample_rate)

    def stft(self, wav):
        D = librosa.core.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode)
        return D

    def istft(self, D):
        wav = librosa.core.istft(
            D,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center)
        return wav

    def spectrogram(self, wav):
        D = self.stft(wav)
        return np.abs(D)

    def mel_spectrogram(self, wav):
        S = self.spectrogram(wav)
        mel = np.dot(self.mel_filter, S)
        return mel
