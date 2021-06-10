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

from typing import Union, Optional, Callable, Tuple, Any
from pathlib import Path
from multiprocessing import Manager

import numpy as np
from paddle.io import Dataset
import logging


class AudioMelDataset(Dataset):
    """Dataset to laod audio and mel dataset.
    
    Parameters
    ----------
    root_dir : Union[Path, str]
        The root of the dataset.
    audio_pattern : str, optional
        A pattern to recursively find all audio files, by default 
        "*-wave.npy"
    mel_pattern : str, optional
        A pattern to recursively find all mel feature files, by default 
        "*-mel.npy"
    audio_load_fn : Callable, optional
        Function to load the audio, which takes a Path object or str as 
        input, by default np.load
    mel_load_fn : Callable, optional
        Function to load the mel features, which takes a Path object or 
        str as input, by default np.load
    audio_length_threshold : Optional[int], optional
        The minmimal length(number of samples) of the audio, by default None
    mel_length_threshold : Optional[int], optional
        The minmimal length(number of frames) of the audio, by default None
    return_utt_id : bool, optional
        Whether to include utterance indentifier in the return value of 
        __getitem__, by default False
    use_cache : bool, optional
        Whether to cache seen examples while reading, by default False
    """

    def __init__(self,
                 root_dir: Union[Path, str],
                 audio_pattern: str="*-wave.npy",
                 mel_pattern: str="*-mel.npy",
                 audio_load_fn: Callable=np.load,
                 mel_load_fn: Callable=np.load,
                 audio_length_threshold: Optional[int]=None,
                 mel_length_threshold: Optional[int]=None,
                 return_utt_id: bool=False,
                 use_cache: bool=False):
        root_dir = Path(root_dir).expanduser()
        # find all of audio and mel files
        audio_files = sorted(list(root_dir.rglob(audio_pattern)))
        mel_files = sorted(list(root_dir.rglob(mel_pattern)))

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [
                idx for idx in range(len(audio_files))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_files) != len(idxs):
                logging.warning(
                    f"Some files are filtered by audio length threshold "
                    f"({len(audio_files)} -> {len(idxs)}).")
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[1] for f in mel_files]
            idxs = [
                idx for idx in range(len(mel_files))
                if mel_lengths[idx] > mel_length_threshold
            ]
            if len(mel_files) != len(idxs):
                logging.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(mel_files)} -> {len(idxs)}).")
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]

        # assert the number of files
        assert len(
            audio_files) != 0, f"Not found any audio files in {root_dir}."
        assert len(audio_files) == len(mel_files), \
            (f"Number of audio and mel files are different "
            f"({len(audio_files)} vs {len(mel_files)}).")

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.mel_load_fn = mel_load_fn
        self.mel_files = mel_files
        if ".npy" in audio_pattern:
            self.utt_ids = [
                f.name.replace("-wave.npy", "") for f in audio_files
            ]
        else:
            self.utt_ids = [f.stem for f in audio_files]
        self.return_utt_id = return_utt_id
        self.use_cache = use_cache
        if use_cache:
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [None for _ in range(len(audio_files))]

    def __getitem__(self, idx: int) -> Tuple:
        """Get an example given the index.

        Parameters
        ----------
        idx : int
            The index of the example.

        Returns
        -------
        utt_id : str
            Utterance identifier.
        audio : np.ndarray
            Shape (n_samples, ), the audio.
        mel: np.ndarray
            Shape (n_mels, n_frames), the mel spectrogram.
        """
        if self.use_cache and self.caches[idx] is not None:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_load_fn(self.audio_files[idx])
        mel = self.mel_load_fn(self.mel_files[idx])

        if self.return_utt_id:
            items = utt_id, audio, mel
        else:
            items = audio, mel

        if self.use_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Returns the size of the dataset.

        Returns
        -------
        int
            The length of the dataset
        """
        return len(self.audio_files)
