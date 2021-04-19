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
import librosa
import pandas
from paddle.io import Dataset, DataLoader


class LJSpeech(Dataset):
    """A simple dataset adaptor for the processed ljspeech dataset."""

    def __init__(self, root, sample_rate, length, top_db):
        self.root = Path(root).expanduser()
        self.metadata = pandas.read_csv(
            str(self.root / "metadata.csv"),
            sep="|",
            header=None,
            names=["fname", "text", "normalized_text"])
        self.wav_dir = self.root / "wavs"
        self.sr = sample_rate
        self.top_db = top_db
        self.length = length # samples in the clip

    def __getitem__(self, i):
        fname = self.metadata.iloc[0].fname
        fpath = (self.wav_dir / fname).with_suffix(".wav")
        y, sr = librosa.load(fpath, self.sr)
        y, _ = librosa.effects.trim(y, top_db=self.top_db)
        y = librosa.util.normalize(y)
        y = y.astype(np.float32)
        
        # pad or trim
        if y.size <= self.length:
            y = np.pad(y, [0, self.length - len(y)], mode='constant')
        else:
            start = np.random.randint(0, 1 + len(y) - self.length)
            y = y[start: start + self.length]
        return y

    def __len__(self):
        return len(self.metadata)


