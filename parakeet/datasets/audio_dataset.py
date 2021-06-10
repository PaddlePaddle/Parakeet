# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Union, Optional, Callable, Tuple
from pathlib import Path
from multiprocessing import Manager

import numpy as np
from paddle.io import Dataset
import logging


class AudioDataset(Dataset):
    """Dataset to load audio.
    
    Parameters
    ----------
    root_dir : Union[Path, str]
        The root of the dataset.
    audio_pattern : str
        A pattern to recursively find all audio files, by default "*-wave.npy"
    audio_length_threshold : int, optional
        The minmimal length(number of samples) of the audio, by default None
    audio_load_fn : Callable, optional
        Function to load the audio, which takes a Path object or str as input, 
        by default np.load
    return_utt_id : bool, optional
        Whether to include utterance indentifier in the return value of 
        __getitem__, by default False
    use_cache : bool, optional
        Whether to cache seen examples while reading, by default False
    """

    def __init__(
            self,
            root_dir: Union[Path, str],
            audio_pattern: str="*-wave.npy",
            audio_length_threshold: Optional[int]=None,
            audio_load_fn: Callable=np.load,
            return_utt_id: bool=False,
            use_cache: bool=False, ):
        # allow str and Path that contains '~'
        root_dir = Path(root_dir).expanduser()

        # recursively find all of audio files that match thr pattern
        audio_files = sorted(list(root_dir.rglob(audio_pattern)))

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [
                idx for idx in range(len(audio_files))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_files) != len(idxs):
                logging.warning(
                    f"some files are filtered by audio length threshold "
                    f"({len(audio_files)} -> {len(idxs)}).")
            audio_files = [audio_files[idx] for idx in idxs]

        # assert the number of files
        assert len(
            audio_files) != 0, f"Not any audio files found in {root_dir}."

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.return_utt_id = return_utt_id
        # TODO(chenfeiyu): better strategy to get utterance id
        if ".npy" in audio_pattern:
            self.utt_ids = [
                f.name.replace("-wave.npy", "") for f in audio_files
            ]
        else:
            self.utt_ids = [f.stem for f in audio_files]
        self.use_cache = use_cache
        if use_cache:
            # use manager to share object between multiple processes
            # avoid per-reader process caching
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [None for _ in range(len(audio_files))]

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray]:
        """Get an example given the index.

        Parameters
        ----------
        idx : int
            The index.

        Returns
        -------
        utt_id : str
            Utterance identifier.
        audio : np.ndarray
            Shape (n_samples, ), the audio.
        """
        if self.use_cache and self.caches[idx] is not None:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_load_fn(self.audio_files[idx])

        if self.return_utt_id:
            items = utt_id, audio
        else:
            items = audio

        if self.use_cache:
            self.caches[idx] = items

        return items

    def __len__(self) -> int:
        """Returns the size of the dataset.

        Returns
        -------
        int
            The length of the dataset
        """
        return len(self.audio_files)
