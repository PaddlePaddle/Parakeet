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
import scipy.io
import scipy.signal


class AudioProcessor(object):
    def __init__(
            self,
            sample_rate=None,  # int, sampling rate
            num_mels=None,  # int, bands of mel spectrogram
            min_level_db=None,  # float, minimum level db
            ref_level_db=None,  # float, reference level db
            n_fft=None,  # int: number of samples in a frame for stft
            win_length=None,  # int: the same meaning with n_fft
            hop_length=None,  # int: number of samples between neighboring frame
            power=None,  # float:power to raise before griffin-lim
            preemphasis=None,  # float: preemphasis coefficident
            signal_norm=None,  # 
            symmetric_norm=False,  # bool, apply clip norm in [-max_norm, max_form]
            max_norm=None,  # float, max norm
            mel_fmin=None,  # int: mel spectrogram's minimum frequency
            mel_fmax=None,  # int: mel spectrogram's maximum frequency
            clip_norm=True,  # bool: clip spectrogram's norm
            griffin_lim_iters=None,  # int:
            do_trim_silence=False,  # bool: trim silence
            sound_norm=False,
            **kwargs):
        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db

        # stft related
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        # hop length defaults to 1/4 window_length
        self.hop_length = hop_length or 0.25 * self.win_length

        self.power = power
        self.preemphasis = float(preemphasis)

        self.griffin_lim_iters = griffin_lim_iters
        self.signal_norm = signal_norm
        self.symmetric_norm = symmetric_norm

        # mel transform related
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax

        self.max_norm = 1.0 if max_norm is None else float(max_norm)
        self.clip_norm = clip_norm
        self.do_trim_silence = do_trim_silence

        self.sound_norm = sound_norm
        self.num_freq, self.frame_length_ms, self.frame_shift_ms = self._stft_parameters(
        )

    def _stft_parameters(self):
        """compute frame length and hop length in ms"""
        frame_length_ms = self.win_length * 1. / self.sample_rate
        frame_shift_ms = self.hop_length * 1. / self.sample_rate
        num_freq = 1 + self.n_fft // 2
        return num_freq, frame_length_ms, frame_shift_ms

    def __repr__(self):
        """object repr"""
        cls_name_str = self.__class__.__name__
        members = vars(self)
        dict_str = "\n".join(
            ["  {}: {},".format(k, v) for k, v in members.items()])
        repr_str = "{}(\n{})\n".format(cls_name_str, dict_str)
        return repr_str

    def save_wav(self, path, wav):
        """save audio with scipy.io.wavfile in 16bit integers"""
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        scipy.io.wavfile.write(path, self.sample_rate,
                               wav_norm.as_type(np.int16))

    def load_wav(self, path, sr=None):
        """load wav -> trim_silence -> rescale"""

        x, sr = librosa.load(path, sr=None)
        assert self.sample_rate == sr, "audio sample rate: {}Hz != processor sample rate: {}Hz".format(
            sr, self.sample_rate)
        if self.do_trim_silence:
            try:
                x = self.trim_silence(x)
            except ValueError:
                print(" [!] File cannot be trimmed for silence - {}".format(
                    path))
        if self.sound_norm:
            x = x / x.max() * 0.9  # why 0.9 ?
        return x

    def trim_silence(self, wav):
        """Trim soilent parts with a threshold and 0.01s margin"""
        margin = int(self.sample_rate * 0.01)
        wav = wav[margin:-margin]
        trimed_wav = librosa.effects.trim(
            wav,
            top_db=60,
            frame_length=self.win_length,
            hop_length=self.hop_length)[0]
        return trimed_wav

    def apply_preemphasis(self, x):
        if self.preemphasis == 0.:
            raise RuntimeError(
                " !! Preemphasis coefficient should be positive. ")
        return scipy.signal.lfilter([1., -self.preemphasis], [1.], x)

    def apply_inv_preemphasis(self, x):
        if self.preemphasis == 0.:
            raise RuntimeError(
                " !! Preemphasis coefficient should be positive. ")
        return scipy.signal.lfilter([1.], [1., -self.preemphasis], x)

    def _amplitude_to_db(self, x):
        amplitude_min = np.exp(self.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(amplitude_min, x))

    @staticmethod
    def _db_to_amplitude(x):
        return np.power(10., 0.05 * x)

    def _linear_to_mel(self, spectrogram):
        _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _mel_to_linear(self, mel_spectrogram):
        inv_mel_basis = np.linalg.pinv(self._build_mel_basis())
        return np.maximum(1e-10, np.dot(inv_mel_basis, mel_spectrogram))

    def _build_mel_basis(self):
        """return mel basis for mel scale"""
        if self.mel_fmax is not None:
            assert self.mel_fmax <= self.sample_rate // 2
        return librosa.filters.mel(self.sample_rate,
                                   self.n_fft,
                                   n_mels=self.num_mels,
                                   fmin=self.mel_fmin,
                                   fmax=self.mel_fmax)

    def _normalize(self, S):
        """put values in [0, self.max_norm] or [-self.max_norm, self,max_norm]"""
        if self.signal_norm:
            S_norm = (S - self.min_level_db) / (-self.min_level_db)
            if self.symmetric_norm:
                S_norm = ((2 * self.max_norm) * S_norm) - self.max_norm
                if self.clip_norm:
                    S_norm = np.clip(S_norm, -self.max_norm, self.max_norm)
                return S_norm
            else:
                S_norm = self.max_norm * S_norm
                if self.clip_norm:
                    S_norm = np.clip(S_norm, 0, self.max_norm)
                return S_norm
        else:
            return S

    def _denormalize(self, S):
        """denormalize values"""
        S_denorm = S
        if self.signal_norm:
            if self.symmetric_norm:
                if self.clip_norm:
                    S_denorm = np.clip(S_denorm, -self.max_norm, self.max_norm)
                S_denorm = (S_denorm + self.max_norm) * (
                    -self.min_level_db) / (2 * self.max_norm
                                           ) + self.min_level_db
                return S_denorm
            else:
                if self.clip_norm:
                    S_denorm = np.clip(S_denorm, 0, self.max_norm)
                S_denorm = S_denorm * (-self.min_level_db
                                       ) / self.max_norm + self.min_level_db
                return S_denorm
        else:
            return S

    def _stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length)

    def _istft(self, S):
        return librosa.istft(
            S, hop_length=self.hop_length, win_length=self.win_length)

    def spectrogram(self, y):
        """compute linear spectrogram(amplitude)
        preemphasis -> stft -> mag -> amplitude_to_db -> minus_ref_level_db -> normalize
        """
        if self.preemphasis:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        S = self._amplitude_to_db(np.abs(D)) - self.ref_level_db
        return self._normalize(S)

    def melspectrogram(self, y):
        """compute linear spectrogram(amplitude)
        preemphasis -> stft -> mag -> mel_scale -> amplitude_to_db -> minus_ref_level_db -> normalize
        """
        if self.preemphasis:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        S = self._amplitude_to_db(self._linear_to_mel(np.abs(
            D))) - self.ref_level_db
        return self._normalize(S)

    def inv_spectrogram(self, spectrogram):
        """convert spectrogram back to waveform using griffin_lim in librosa"""
        S = self._denormalize(spectrogram)
        S = self._db_to_amplitude(S + self.ref_level_db)
        if self.preemphasis:
            return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        return self._griffin_lim(S**self.power)

    def inv_melspectrogram(self, mel_spectrogram):
        S = self._denormalize(mel_spectrogram)
        S = self._db_to_amplitude(S + self.ref_level_db)
        S = self._mel_to_linear(np.abs(S))
        if self.preemphasis:
            return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        return self._griffin_lim(S**self.power)

    def out_linear_to_mel(self, linear_spec):
        """convert output linear spec to mel spec"""
        S = self._denormalize(linear_spec)
        S = self._db_to_amplitude(S + self.ref_level_db)
        S = self._linear_to_mel(np.abs(S))
        S = self._amplitude_to_db(S) - self.ref_level_db
        mel = self._normalize(S)
        return mel

    def _griffin_lim(self, S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for _ in range(self.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    @staticmethod
    def mulaw_encode(wav, qc):
        mu = 2**qc - 1
        # wav_abs = np.minimum(np.abs(wav), 1.0)
        signal = np.sign(wav) * np.log(1 + mu * np.abs(wav)) / np.log(1. + mu)
        # Quantize signal to the specified number of levels.
        signal = (signal + 1) / 2 * mu + 0.5
        return np.floor(signal, )

    @staticmethod
    def mulaw_decode(wav, qc):
        """Recovers waveform from quantized values."""
        mu = 2**qc - 1
        x = np.sign(wav) / mu * ((1 + mu)**np.abs(wav) - 1)
        return x

    @staticmethod
    def encode_16bits(x):
        return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)

    @staticmethod
    def quantize(x, bits):
        return (x + 1.) * (2**bits - 1) / 2

    @staticmethod
    def dequantize(x, bits):
        return 2 * x / (2**bits - 1) - 1
