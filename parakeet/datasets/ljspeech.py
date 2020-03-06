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
import numpy as np
import pandas as pd
import librosa
from .. import g2p

from ..data.sampler import SequentialSampler, RandomSampler, BatchSampler
from ..data.dataset import DatasetMixin
from ..data.datacargo import DataCargo
from ..data.batch import TextIDBatcher, SpecBatcher


class LJSpeech(DatasetMixin):
    def __init__(self, root):
        super(LJSpeech, self).__init__()
        self.root = root
        self.metadata = self._prepare_metadata()

    def _prepare_metadata(self):
        csv_path = os.path.join(self.root, "metadata.csv")
        metadata = pd.read_csv(
            csv_path,
            sep="|",
            header=None,
            quoting=3,
            names=["fname", "raw_text", "normalized_text"])
        return metadata

    def _get_example(self, metadatum):
        """All the code for generating an Example from a metadatum. If you want a 
        different preprocessing pipeline, you can override this method. 
        This method may require several processor, each of which has a lot of options.
        In this case, you'd better pass a composed transform and pass it to the init
        method.
        """

        fname, raw_text, normalized_text = metadatum
        wav_path = os.path.join(self.root, "wavs", fname + ".wav")

        # load -> trim -> preemphasis -> stft -> magnitude -> mel_scale -> logscale -> normalize
        wav, sample_rate = librosa.load(
            wav_path,
            sr=None)  # we would rather use functor to hold its parameters
        trimed, _ = librosa.effects.trim(wav)
        preemphasized = librosa.effects.preemphasis(trimed)
        D = librosa.stft(preemphasized)
        mag, phase = librosa.magphase(D)
        mel = librosa.feature.melspectrogram(S=mag)

        mag = librosa.amplitude_to_db(S=mag)
        mel = librosa.amplitude_to_db(S=mel)

        ref_db = 20
        max_db = 100
        mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
        mel = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

        phonemes = np.array(
            g2p.en.text_to_sequence(normalized_text), dtype=np.int64)
        return (mag, mel, phonemes
                )  # maybe we need to implement it as a map in the future

    def _batch_examples(self, minibatch):
        mag_batch = []
        mel_batch = []
        phoneme_batch = []
        for example in minibatch:
            mag, mel, phoneme = example
            mag_batch.append(mag)
            mel_batch.append(mel)
            phoneme_batch.append(phoneme)
        mag_batch = SpecBatcher(pad_value=0.)(mag_batch)
        mel_batch = SpecBatcher(pad_value=0.)(mel_batch)
        phoneme_batch = TextIDBatcher(pad_id=0)(phoneme_batch)
        return (mag_batch, mel_batch, phoneme_batch)

    def __getitem__(self, index):
        metadatum = self.metadata.iloc[index]
        example = self._get_example(metadatum)
        return example

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.metadata)
