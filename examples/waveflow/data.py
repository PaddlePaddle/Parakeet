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

import os
import random

import librosa
import numpy as np
from paddle import fluid

from parakeet.datasets import ljspeech
from parakeet.data import SpecBatcher, WavBatcher
from parakeet.data import DataCargo, DatasetMixin
from parakeet.data import DistributedSampler, BatchSampler
from scipy.io.wavfile import read


class Dataset(ljspeech.LJSpeech):
    def __init__(self, config):
        super(Dataset, self).__init__(config.root)
        self.config = config

    def _get_example(self, metadatum):
        fname, _, _ = metadatum
        wav_path = os.path.join(self.root, "wavs", fname + ".wav")

        loaded_sr, audio = read(wav_path)
        assert loaded_sr == self.config.sample_rate

        return audio


class Subset(DatasetMixin):
    def __init__(self, dataset, indices, valid):
        self.dataset = dataset
        self.indices = indices
        self.valid = valid
        self.config = dataset.config

    def get_mel(self, audio):
        spectrogram = librosa.core.stft(
            audio,
            n_fft=self.config.fft_size,
            hop_length=self.config.fft_window_shift,
            win_length=self.config.fft_window_size)
        spectrogram_magnitude = np.abs(spectrogram)

        # mel_filter_bank shape: [n_mels, 1 + n_fft/2]
        mel_filter_bank = librosa.filters.mel(sr=self.config.sample_rate,
                                              n_fft=self.config.fft_size,
                                              n_mels=self.config.mel_bands,
                                              fmin=self.config.mel_fmin,
                                              fmax=self.config.mel_fmax)
        # mel shape: [n_mels, num_frames]
        mel = np.dot(mel_filter_bank, spectrogram_magnitude)

        # Normalize mel.
        clip_val = 1e-5
        ref_constant = 1
        mel = np.log(np.clip(mel, a_min=clip_val, a_max=None) * ref_constant)

        return mel

    def __getitem__(self, idx):
        audio = self.dataset[self.indices[idx]]
        segment_length = self.config.segment_length

        if self.valid:
            # whole audio for valid set
            pass
        else:
            # Randomly crop segment_length from audios in the training set.
            # audio shape: [len]
            if audio.shape[0] >= segment_length:
                max_audio_start = audio.shape[0] - segment_length
                audio_start = random.randint(0, max_audio_start)
                audio = audio[audio_start:(audio_start + segment_length)]
            else:
                audio = np.pad(audio, (0, segment_length - audio.shape[0]),
                               mode='constant',
                               constant_values=0)

        # Normalize audio to the [-1, 1] range.
        audio = audio.astype(np.float32) / 32768.0
        mel = self.get_mel(audio)

        return audio, mel

    def _batch_examples(self, batch):
        audios = [sample[0] for sample in batch]
        mels = [sample[1] for sample in batch]

        audios = WavBatcher(pad_value=0.0)(audios)
        mels = SpecBatcher(pad_value=0.0)(mels)

        return audios, mels

    def __len__(self):
        return len(self.indices)


class LJSpeech:
    def __init__(self, config, nranks, rank):
        place = fluid.CUDAPlace(rank) if config.use_gpu else fluid.CPUPlace()

        # Whole LJSpeech dataset.
        ds = Dataset(config)

        # Split into train and valid dataset.
        indices = list(range(len(ds)))
        train_indices = indices[config.valid_size:]
        valid_indices = indices[:config.valid_size]
        random.shuffle(train_indices)

        # Train dataset.
        trainset = Subset(ds, train_indices, valid=False)
        sampler = DistributedSampler(len(trainset), nranks, rank)
        total_bs = config.batch_size
        assert total_bs % nranks == 0
        train_sampler = BatchSampler(
            sampler, total_bs // nranks, drop_last=True)
        trainloader = DataCargo(trainset, batch_sampler=train_sampler)

        trainreader = fluid.io.PyReader(capacity=50, return_list=True)
        trainreader.decorate_batch_generator(trainloader, place)
        self.trainloader = (data for _ in iter(int, 1)
                            for data in trainreader())

        # Valid dataset.
        validset = Subset(ds, valid_indices, valid=True)
        # Currently only support batch_size = 1 for valid loader.
        validloader = DataCargo(validset, batch_size=1, shuffle=False)

        validreader = fluid.io.PyReader(capacity=20, return_list=True)
        validreader.decorate_batch_generator(validloader, place)
        self.validloader = validreader
