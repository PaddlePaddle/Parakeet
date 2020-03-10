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
import csv
import numpy as np
import librosa
from pathlib import Path
import pandas as pd

from parakeet.data import batch_spec, batch_wav
from parakeet.data import DatasetMixin


class LJSpeechMetaData(DatasetMixin):
    def __init__(self, root):
        self.root = Path(root)
        self._wav_dir = self.root.joinpath("wavs")
        csv_path = self.root.joinpath("metadata.csv")
        self._table = pd.read_csv(
            csv_path,
            sep="|",
            header=None,
            quoting=csv.QUOTE_NONE,
            names=["fname", "raw_text", "normalized_text"])

    def get_example(self, i):
        fname, raw_text, normalized_text = self._table.iloc[i]
        fname = str(self._wav_dir.joinpath(fname + ".wav"))
        return fname, raw_text, normalized_text

    def __len__(self):
        return len(self._table)


class Transform(object):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __call__(self, example):
        wav_path, _, _ = example

        sr = self.sample_rate
        n_fft = self.n_fft
        win_length = self.win_length
        hop_length = self.hop_length
        n_mels = self.n_mels

        wav, loaded_sr = librosa.load(wav_path, sr=None)
        assert loaded_sr == sr, "sample rate does not match, resampling applied"

        # Pad audio to the right size.
        frames = int(np.ceil(float(wav.size) / hop_length))
        fft_padding = (n_fft - hop_length) // 2  # sound
        desired_length = frames * hop_length + fft_padding * 2
        pad_amount = (desired_length - wav.size) // 2

        if wav.size % 2 == 0:
            wav = np.pad(wav, (pad_amount, pad_amount), mode='reflect')
        else:
            wav = np.pad(wav, (pad_amount, pad_amount + 1), mode='reflect')

        # Normalize audio.
        wav = wav / np.abs(wav).max() * 0.999

        # Compute mel-spectrogram.
        # Turn center to False to prevent internal padding.
        spectrogram = librosa.core.stft(
            wav,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            center=False)
        spectrogram_magnitude = np.abs(spectrogram)

        # Compute mel-spectrograms.
        mel_filter_bank = librosa.filters.mel(sr=sr,
                                              n_fft=n_fft,
                                              n_mels=n_mels)
        mel_spectrogram = np.dot(mel_filter_bank, spectrogram_magnitude)
        mel_spectrogram = mel_spectrogram

        # Rescale mel_spectrogram.
        min_level, ref_level = 1e-5, 20  # hard code it
        mel_spectrogram = 20 * np.log10(np.maximum(min_level, mel_spectrogram))
        mel_spectrogram = mel_spectrogram - ref_level
        mel_spectrogram = np.clip((mel_spectrogram + 100) / 100, 0, 1)

        # Extract the center of audio that corresponds to mel spectrograms.
        audio = wav[fft_padding:-fft_padding]
        assert mel_spectrogram.shape[1] * hop_length == audio.size

        # there is no clipping here
        return audio, mel_spectrogram


class DataCollector(object):
    def __init__(self,
                 context_size,
                 sample_rate,
                 hop_length,
                 train_clip_seconds,
                 valid=False):
        frames_per_second = sample_rate // hop_length
        train_clip_frames = int(
            np.ceil(train_clip_seconds * frames_per_second))
        context_frames = context_size // hop_length
        self.num_frames = train_clip_frames + context_frames

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.valid = valid

    def random_crop(self, sample):
        audio, mel_spectrogram = sample
        audio_frames = int(audio.size) // self.hop_length
        max_start_frame = audio_frames - self.num_frames
        assert max_start_frame >= 0, "audio is too short to be cropped"

        frame_start = np.random.randint(0, max_start_frame)
        # frame_start = 0  # norandom
        frame_end = frame_start + self.num_frames

        audio_start = frame_start * self.hop_length
        audio_end = frame_end * self.hop_length

        audio = audio[audio_start:audio_end]
        return audio, mel_spectrogram, audio_start

    def __call__(self, samples):
        # transform them first
        if self.valid:
            samples = [(audio, mel_spectrogram, 0)
                       for audio, mel_spectrogram in samples]
        else:
            samples = [self.random_crop(sample) for sample in samples]
        # batch them
        audios = [sample[0] for sample in samples]
        audio_starts = [sample[2] for sample in samples]
        mels = [sample[1] for sample in samples]

        mels = batch_spec(mels)

        if self.valid:
            audios = batch_wav(audios, dtype=np.float32)
        else:
            audios = np.array(audios, dtype=np.float32)
        audio_starts = np.array(audio_starts, dtype=np.int64)
        return audios, mels, audio_starts
