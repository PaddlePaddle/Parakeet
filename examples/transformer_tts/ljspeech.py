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
from paddle.io import Dataset, DataLoader

from parakeet.data.batch import batch_spec, batch_text_id
from parakeet.data import dataset


class LJSpeech(Dataset):
    """A simple dataset adaptor for the processed ljspeech dataset."""

    def __init__(self, root):
        self.root = Path(root).expanduser()
        records = []
        with open(self.root / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        for mel_name, text, phonemes, ids in metadata:
            mel_name = self.root / "mel" / (mel_name + ".npy")
            records.append((mel_name, text, phonemes, ids))
        self.records = records

    def __getitem__(self, i):
        mel_name, _, _, ids = self.records[i]
        mel = np.load(mel_name)
        return ids, mel

    def __len__(self):
        return len(self.records)


# decorate mel & create stop probability
class Transform(object):
    def __init__(self, start_value, end_value):
        self.start_value = start_value
        self.end_value = end_value

    def __call__(self, example):
        ids, mel = example  # ids already have <s> and </s>
        ids = np.array(ids, dtype=np.int64)
        # add start and end frame
        mel = np.pad(
            mel, [(0, 0), (1, 1)],
            mode='constant',
            constant_values=[(0, 0), (self.start_value, self.end_value)])
        stop_labels = np.ones([mel.shape[1]], dtype=np.int64)
        stop_labels[-1] = 2
        # actually this thing can also be done within the model
        return ids, mel, stop_labels


class LJSpeechCollector(object):
    """A simple callable to batch LJSpeech examples."""

    def __init__(self, padding_idx=0, padding_value=0.):
        self.padding_idx = padding_idx
        self.padding_value = padding_value

    def __call__(self, examples):
        ids = [example[0] for example in examples]
        mels = [example[1] for example in examples]
        stop_probs = [example[2] for example in examples]

        ids, _ = batch_text_id(ids, pad_id=self.padding_idx)
        mels, _ = batch_spec(mels, pad_value=self.padding_value)
        stop_probs, _ = batch_text_id(stop_probs, pad_id=self.padding_idx)
        return ids, np.transpose(mels, [0, 2, 1]), stop_probs


def create_dataloader(config, source_path):
    lj = LJSpeech(source_path)
    transform = Transform(config.data.mel_start_value,
                          config.data.mel_end_value)
    lj = dataset.TransformDataset(lj, transform)

    valid_set, train_set = dataset.split(lj, config.data.valid_size)
    data_collator = LJSpeechCollector(padding_idx=config.data.padding_idx)
    train_loader = DataLoader(
        train_set,
        batch_size=config.data.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=data_collator)
    valid_loader = DataLoader(
        valid_set,
        batch_size=config.data.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=data_collator)
    return train_loader, valid_loader
