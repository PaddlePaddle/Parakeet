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

from pathlib import Path
import pandas as pd
from ruamel.yaml import YAML
import io

import librosa
import numpy as np

from parakeet.g2p.en import text_to_sequence
from parakeet.data.dataset import Dataset
from parakeet.data.datacargo import DataCargo
from parakeet.data.batch import TextIDBatcher, WavBatcher


class VCTK(Dataset):
    def __init__(self, root):
        assert isinstance(root, (
            str, Path)), "root should be a string or Path object"
        self.root = root if isinstance(root, Path) else Path(root)
        self.text_root = self.root.joinpath("txt")
        self.wav_root = self.root.joinpath("wav48")

        if not (self.root.joinpath("metadata.csv").exists() and
                self.root.joinpath("speaker_indices.yaml").exists()):
            self._prepare_metadata()
        self.speaker_indices, self.metadata = self._load_metadata()

    def _load_metadata(self):
        yaml = YAML(typ='safe')
        speaker_indices = yaml.load(self.root.joinpath("speaker_indices.yaml"))
        metadata = pd.read_csv(
            self.root.joinpath("metadata.csv"), sep="|", quoting=3, header=1)
        return speaker_indices, metadata

    def _prepare_metadata(self):
        metadata = []
        speaker_to_index = {}
        for i, speaker_folder in enumerate(self.text_root.iterdir()):
            if speaker_folder.is_dir():
                speaker_to_index[speaker_folder.name] = i
                for text_file in speaker_folder.iterdir():
                    if text_file.is_file():
                        with io.open(str(text_file)) as f:
                            transcription = f.read().strip()
                    wav_file = text_file.with_suffix(".wav")
                    metadata.append(
                        (wav_file.name, speaker_folder.name, transcription))
        metadata = pd.DataFrame.from_records(
            metadata, columns=["wave_file", "speaker", "text"])

        # save them
        yaml = YAML(typ='safe')
        yaml.dump(speaker_to_index, self.root.joinpath("speaker_indices.yaml"))
        metadata.to_csv(
            self.root.joinpath("metadata.csv"),
            sep="|",
            quoting=3,
            index=False)

    def _get_example(self, metadatum):
        wave_file, speaker, text = metadatum
        wav_path = self.wav_root.joinpath(speaker, wave_file)
        wav, sr = librosa.load(str(wav_path), sr=None)
        phoneme_seq = np.array(text_to_sequence(text))
        return wav, self.speaker_indices[speaker], phoneme_seq

    def __getitem__(self, index):
        metadatum = self.metadata.iloc[index]
        example = self._get_example(metadatum)
        return example

    def __len__(self):
        return len(self.metadata)

    def _batch_examples(self, minibatch):
        wav_batch, speaker_batch, phoneme_batch = [], [], []
        for example in minibatch:
            wav, speaker_id, phoneme_seq = example
            wav_batch.append(wav)
            speaker_batch.append(speaker_id)
            phoneme_batch.append(phoneme_seq)
        wav_batch = WavBatcher(pad_value=0.)(wav_batch)
        speaker_batch = np.array(speaker_batch)
        phoneme_batch = TextIDBatcher(pad_id=0)(phoneme_batch)
        return wav_batch, speaker_batch, phoneme_batch
