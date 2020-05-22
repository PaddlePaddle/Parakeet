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
import os
import csv
from pathlib import Path
import numpy as np
from paddle import fluid
import pandas as pd
import librosa
from scipy import signal

import paddle.fluid.dygraph as dg

from parakeet.g2p.en import text_to_sequence, sequence_to_text
from parakeet.data import DatasetMixin, TransformDataset, FilterDataset, CacheDataset
from parakeet.data import DataCargo, PartialyRandomizedSimilarTimeLengthSampler, SequentialSampler, BucketSampler


class LJSpeechMetaData(DatasetMixin):
    def __init__(self, root):
        self.root = Path(root)
        self._wav_dir = self.root.joinpath("wavs")
        csv_path = self.root.joinpath("metadata.csv")
        self._table = pd.read_csv(
            csv_path,
            sep="|",
            encoding="utf-8",
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
    def __init__(self,
                 replace_pronunciation_prob=0.,
                 sample_rate=22050,
                 preemphasis=.97,
                 n_fft=1024,
                 win_length=1024,
                 hop_length=256,
                 fmin=125,
                 fmax=7600,
                 n_mels=80,
                 min_level_db=-100,
                 ref_level_db=20,
                 max_norm=0.999,
                 clip_norm=True):
        self.replace_pronunciation_prob = replace_pronunciation_prob

        self.sample_rate = sample_rate
        self.preemphasis = preemphasis
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.fmin = fmin
        self.fmax = fmax
        self.n_mels = n_mels

        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db
        self.max_norm = max_norm
        self.clip_norm = clip_norm

    def __call__(self, in_data):
        fname, _, normalized_text = in_data

        # text processing
        mix_grapheme_phonemes = text_to_sequence(
            normalized_text, self.replace_pronunciation_prob)
        text_length = len(mix_grapheme_phonemes)
        # CAUTION: positions start from 1
        speaker_id = None

        # wave processing
        wav, _ = librosa.load(fname, sr=self.sample_rate)
        # preemphasis
        y = signal.lfilter([1., -self.preemphasis], [1.], wav)

        # STFT
        D = librosa.stft(
            y=y,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length)
        S = np.abs(D)

        # to db and normalize to 0-1
        amplitude_min = np.exp(self.min_level_db / 20 * np.log(10))  # 1e-5
        S_norm = 20 * np.log10(np.maximum(amplitude_min,
                                          S)) - self.ref_level_db
        S_norm = (S_norm - self.min_level_db) / (-self.min_level_db)
        S_norm = self.max_norm * S_norm
        if self.clip_norm:
            S_norm = np.clip(S_norm, 0, self.max_norm)

        # mel scale and to db and normalize to 0-1,
        # CAUTION: pass linear scale S, not dbscaled S
        S_mel = librosa.feature.melspectrogram(
            S=S, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, power=1.)
        S_mel = 20 * np.log10(np.maximum(amplitude_min,
                                         S_mel)) - self.ref_level_db
        S_mel_norm = (S_mel - self.min_level_db) / (-self.min_level_db)
        S_mel_norm = self.max_norm * S_mel_norm
        if self.clip_norm:
            S_mel_norm = np.clip(S_mel_norm, 0, self.max_norm)

        # num_frames
        n_frames = S_mel_norm.shape[-1]  # CAUTION: original number of frames
        return (mix_grapheme_phonemes, text_length, speaker_id, S_norm.T,
                S_mel_norm.T, n_frames)


class DataCollector(object):
    def __init__(self, downsample_factor=4, r=1):
        self.downsample_factor = int(downsample_factor)
        self.frames_per_step = int(r)
        self._factor = int(downsample_factor * r)
        # CAUTION: small diff here
        self._pad_begin = int(downsample_factor * r)

    def __call__(self, examples):
        batch_size = len(examples)

        # lengths
        text_lengths = np.array([example[1]
                                 for example in examples]).astype(np.int64)
        frames = np.array([example[5]
                           for example in examples]).astype(np.int64)

        max_text_length = int(np.max(text_lengths))
        max_frames = int(np.max(frames))
        if max_frames % self._factor != 0:
            max_frames += (self._factor - max_frames % self._factor)
        max_frames += self._pad_begin
        max_decoder_length = max_frames // self._factor

        # pad time sequence
        text_sequences = []
        lin_specs = []
        mel_specs = []
        done_flags = []
        for example in examples:
            (mix_grapheme_phonemes, text_length, speaker_id, S_norm,
             S_mel_norm, num_frames) = example
            text_sequences.append(
                np.pad(mix_grapheme_phonemes, (0, max_text_length - text_length
                                               ),
                       mode="constant"))
            lin_specs.append(
                np.pad(S_norm, ((self._pad_begin, max_frames - self._pad_begin
                                 - num_frames), (0, 0)),
                       mode="constant"))
            mel_specs.append(
                np.pad(S_mel_norm, ((self._pad_begin, max_frames -
                                     self._pad_begin - num_frames), (0, 0)),
                       mode="constant"))
            done_flags.append(
                np.pad(np.zeros((int(np.ceil(num_frames // self._factor)), )),
                       (0, max_decoder_length - int(
                           np.ceil(num_frames // self._factor))),
                       mode="constant",
                       constant_values=1))
        text_sequences = np.array(text_sequences).astype(np.int64)
        lin_specs = np.array(lin_specs).astype(np.float32)
        mel_specs = np.array(mel_specs).astype(np.float32)

        # downsample here
        done_flags = np.array(done_flags).astype(np.float32)

        # text positions
        text_mask = (np.arange(1, 1 + max_text_length) <= np.expand_dims(
            text_lengths, -1)).astype(np.int64)
        text_positions = np.arange(
            1, 1 + max_text_length, dtype=np.int64) * text_mask

        # decoder_positions
        decoder_positions = np.tile(
            np.expand_dims(
                np.arange(
                    1, 1 + max_decoder_length, dtype=np.int64), 0),
            (batch_size, 1))

        return (text_sequences, text_lengths, text_positions, mel_specs,
                lin_specs, frames, decoder_positions, done_flags)


def make_data_loader(data_root, config):
    # construct meta data
    meta = LJSpeechMetaData(data_root)

    # filter it!
    min_text_length = config["meta_data"]["min_text_length"]
    meta = FilterDataset(meta, lambda x: len(x[2]) >= min_text_length)

    # transform meta data into meta data
    c = config["transform"]
    transform = Transform(
        replace_pronunciation_prob=c["replace_pronunciation_prob"],
        sample_rate=c["sample_rate"],
        preemphasis=c["preemphasis"],
        n_fft=c["n_fft"],
        win_length=c["win_length"],
        hop_length=c["hop_length"],
        fmin=c["fmin"],
        fmax=c["fmax"],
        n_mels=c["n_mels"],
        min_level_db=c["min_level_db"],
        ref_level_db=c["ref_level_db"],
        max_norm=c["max_norm"],
        clip_norm=c["clip_norm"])
    ljspeech = CacheDataset(TransformDataset(meta, transform))

    # use meta data's text length as a sort key for the sampler
    batch_size = config["train"]["batch_size"]
    text_lengths = [len(example[2]) for example in meta]
    sampler = PartialyRandomizedSimilarTimeLengthSampler(text_lengths,
                                                         batch_size)

    env = dg.parallel.ParallelEnv()
    num_trainers = env.nranks
    local_rank = env.local_rank
    sampler = BucketSampler(
        text_lengths, batch_size, num_trainers=num_trainers, rank=local_rank)

    # some model hyperparameters affect how we process data
    model_config = config["model"]
    collector = DataCollector(
        downsample_factor=model_config["downsample_factor"],
        r=model_config["outputs_per_step"])
    ljspeech_loader = DataCargo(
        ljspeech, batch_fn=collector, batch_size=batch_size, sampler=sampler)
    loader = fluid.io.DataLoader.from_generator(capacity=10, return_list=True)
    loader.set_batch_generator(
        ljspeech_loader, places=fluid.framework._current_expected_place())
    return loader
