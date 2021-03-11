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
import numpy as np
import librosa
from scipy.signal import lfilter


def label_2_float(x, bits):
    return 2 * x / (2 ** bits - 1.) - 1.


def float_2_label(x, bits):
    assert abs(x).max() <= 1.0
    x = (x + 1.) * (2 ** bits - 1) / 2
    return x.clip(0, 2 ** bits - 1)


def load_wav(path, sr):
    return librosa.load(path, sr=sr)[0]


def save_wav(x, path, sr):
    librosa.output.write_wav(path, x.astype(np.float32), sr=sr)


def split_signal(x):
    unsigned = x + 2 ** 15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine


def combine_signal(coarse, fine):
    return coarse * 256 + fine - 2 ** 15


def encode_16bits(x):
    return np.clip(x * (2 ** 15), -2 ** 15, 2 ** 15 - 1).astype(np.int16)


def linear_to_mel(spectrogram, sr, n_fft, num_mels, fmin, fmax):
    return librosa.feature.melspectrogram(
        S=spectrogram, sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)


def normalize(S, min_level_db):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def denormalize(S, min_level_db):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def spectrogram(y, n_fft, ref_level_db, min_level_db, hop_length, win_length):
    # TODO
    D = stft(y, n_fft, hop_length, win_length)
    S = amp_to_db(np.abs(D)) - ref_level_db
    return normalize(S=S, min_level_db=min_level_db)


def melspectrogram(y, sr, n_fft, num_mels, fmin, fmax, ref_level_db, min_level_db, hop_length, win_length):
    # TODO
    D = stft(y, n_fft, hop_length, win_length)
    S = amp_to_db(linear_to_mel(np.abs(D), sr, n_fft, num_mels, fmin, fmax))
    return normalize(S, min_level_db)


def stft(y, n_fft, hop_length, win_length):
    return librosa.stft(
        y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length
    )


def pre_emphasis(x, preemphasis):
    return lfilter([1, -preemphasis], [1], x)


def de_emphasis(x, preemphasis):
    return lfilter([1], [1, -preemphasis], x)


def encode_mu_law(x, mu):
    # mu = 2 ** 9 = 512
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def decode_mu_law(y, mu, from_labels=True):
    # TODO: get rid of log2 - makes no sense
    if from_labels:
        y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def reconstruct_waveform(mel, sr, n_fft, hop_length, win_length, fmin, fmax, min_level_db, n_iter=32):
    """
    Uses Griffin-Lim phase reconstruction to convert from a normalized
    mel spectrogram back into a waveform.
    """
    denormalized = denormalize(mel, min_level_db)
    amp_mel = db_to_amp(denormalized)
    S = librosa.feature.inverse.mel_to_stft(
        amp_mel, power=1, sr=sr, n_fft=n_fft, fmin=fmin, fmax=fmax)
    wav = librosa.core.griffinlim(
        S, n_iter=n_iter,
        hop_length=hop_length, win_length=win_length)
    return wav



















