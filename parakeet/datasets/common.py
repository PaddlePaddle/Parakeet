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

from paddle.io import Dataset
import os
import librosa

__all__ = ["AudioFolderDataset"]


class AudioFolderDataset(Dataset):
    def __init__(self, path, sample_rate, extension="wav"):
        self.root = os.path.expanduser(path)
        self.sample_rate = sample_rate
        self.extension = extension
        self.file_names = [
            os.path.join(self.root, x) for x in os.listdir(self.root) \
                if os.path.splitext(x)[-1] == self.extension]
        self.length = len(self.file_names)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        file_name = self.file_names[i]
        y, _ = librosa.load(file_name, sr=self.sample_rate)  # pylint: disable=unused-variable
        return y
