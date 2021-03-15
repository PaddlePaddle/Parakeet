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
from pathlib import Path
import pickle
import numpy as np
import pandas
from paddle.io import Dataset, DataLoader
import paddle
from parakeet.data.batch import batch_spec, batch_wav
from parakeet.data import dataset
from parakeet.audio import AudioProcessor
from utils.audio import label_2_float, float_2_label

from config import get_cfg_defaults


class Baker(Dataset):
    """A simple dataset adaptor for the processed ljspeech dataset."""

    def __init__(self, root):
        self.root = Path(root).expanduser()
        meta_data = pandas.read_csv(
            str(self.root / "metadata.csv"),
            sep="\t",
            header=None,
            names=["fname", "frames", "samples"])

        records = []
        for row in meta_data.itertuples():
            mel_path = str(self.root / "mel" / (str(row.fname) + ".npy"))
            wav_path = str(self.root / "wav" / (str(row.fname) + ".npy"))
            records.append((mel_path, wav_path))
        self.records = records

    def __getitem__(self, i):
        mel_name, wav_name = self.records[i]
        mel = np.load(mel_name)
        wav = np.load(wav_name)
        return mel, wav

    def __len__(self):
        return len(self.records)


class BakerCollate(object):
    def __init__(self, mode, seq_len, hop_length, pad, bits):
        self.mode = mode
        self.mel_win = seq_len // hop_length + 2 * pad
        self.seq_len = seq_len
        self.hop_length = hop_length
        self.pad = pad
        if self.mode == 'MOL':
            self.bits = 16
        else:
            self.bits = bits

    def __call__(self, batch):
        # batch: [mel, quant]
        # voc_pad = 2  this will pad the input so that the resnet can 'see' wider than input length
        # max_offsets = n_frames - 2 - (mel_win + 2 * hp.voc_pad) = n_frames - 15
        max_offsets = [x[0].shape[-1] - 2 - (self.mel_win + 2 * self.pad) for x in batch]
        # the slice point of mel selecting randomly 
        mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
        # the slice point of wav selecting randomly, which is behind 2(=pad) frames 
        sig_offsets = [(offset + self.pad) * self.hop_length for offset in mel_offsets]
        # mels.sape[1] = voc_seq_len // hop_length + 2 * voc_pad
        mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + self.mel_win] for i, x in enumerate(batch)]
        # label.shape[1] = voc_seq_len + 1
        labels = [x[1][sig_offsets[i]:sig_offsets[i] + self.seq_len + 1] for i, x in enumerate(batch)]

        mels = np.stack(mels).astype(np.float32)
        labels = np.stack(labels).astype(np.int64)

        mels = paddle.to_tensor(mels)
        labels = paddle.to_tensor(labels, dtype='int64')

        # x is input, y is label
        x = labels[:, :self.seq_len]
        y = labels[:, 1:]
        '''
            mode = RAW:
                mu_law = True:
                    quant: bits = 9   0, 1, 2, ..., 509, 510, 511  int
                mu_law = False
                    quant bits = 9    [0， 511]  float
            mode = MOL:
                quant: bits = 16  [0. 65536]  float
        '''
        # x should be normalizes in.[0, 1] in RAW mode
        x = label_2_float(paddle.cast(x, dtype='float32'), self.bits)
        # y should be normalizes in.[0, 1] in MOL mode
        if self.mode == 'MOL':
            y = label_2_float(paddle.cast(y, dtype='float32'), self.bits)

        return x, y, mels

