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
"""
functions to make batch for arrays which satisfy some conditions.
"""
import numpy as np


class TextIDBatcher(object):
    """A wrapper class for a function to build a functor, which holds the configs to pass to the function."""

    def __init__(self, pad_id=0, dtype=np.int64):
        self.pad_id = pad_id
        self.dtype = dtype

    def __call__(self, minibatch):
        out = batch_text_id(minibatch, pad_id=self.pad_id, dtype=self.dtype)
        return out


def batch_text_id(minibatch, pad_id=0, dtype=np.int64):
    """
    minibatch: List[Example]
    Example: ndarray, shape(T,), dtype: int64
    """
    peek_example = minibatch[0]
    assert len(peek_example.shape) == 1, "text example is an 1D tensor"

    lengths = [example.shape[0] for example in minibatch
               ]  # assume (channel, n_samples) or (n_samples, )
    max_len = np.max(lengths)

    batch = []
    for example in minibatch:
        pad_len = max_len - example.shape[0]
        batch.append(
            np.pad(example, [(0, pad_len)],
                   mode='constant',
                   constant_values=pad_id))

    return np.array(batch, dtype=dtype)


class WavBatcher(object):
    def __init__(self, pad_value=0., dtype=np.float32):
        self.pad_value = pad_value
        self.dtype = dtype

    def __call__(self, minibatch):
        out = batch_wav(minibatch, pad_value=self.pad_value, dtype=self.dtype)
        return out


def batch_wav(minibatch, pad_value=0., dtype=np.float32):
    """
    minibatch: List[Example]
    Example: ndarray, shape(C, T) for multi-channel wav, shape(T,) for mono-channel wav, dtype: float32 
    """
    # detect data format, maybe better to specify it in __init__
    peek_example = minibatch[0]
    if len(peek_example.shape) == 1:
        mono_channel = True
    elif len(peek_example.shape) == 2:
        mono_channel = False

    lengths = [example.shape[-1] for example in minibatch
               ]  # assume (channel, n_samples) or (n_samples, )
    max_len = np.max(lengths)

    batch = []
    for example in minibatch:
        pad_len = max_len - example.shape[-1]
        if mono_channel:
            batch.append(
                np.pad(example, [(0, pad_len)],
                       mode='constant',
                       constant_values=pad_value))
        else:
            batch.append(
                np.pad(example, [(0, 0), (0, pad_len)],
                       mode='constant',
                       constant_values=pad_value))  # what about PCM, no

    return np.array(batch, dtype=dtype)


class SpecBatcher(object):
    def __init__(self, pad_value=0., dtype=np.float32):
        self.pad_value = pad_value
        self.dtype = dtype

    def __call__(self, minibatch):
        out = batch_spec(minibatch, pad_value=self.pad_value, dtype=self.dtype)
        return out


def batch_spec(minibatch, pad_value=0., dtype=np.float32):
    """
    minibatch: List[Example]
    Example: ndarray, shape(C, F, T) for multi-channel spectrogram, shape(F, T) for mono-channel spectrogram, dtype: float32 
    """
    # assume (F, T) or (C, F, T)
    peek_example = minibatch[0]
    if len(peek_example.shape) == 2:
        mono_channel = True
    elif len(peek_example.shape) == 3:
        mono_channel = False

    lengths = [example.shape[-1] for example in minibatch
               ]  # assume (channel, F, n_frame) or (F, n_frame)
    max_len = np.max(lengths)

    batch = []
    for example in minibatch:
        pad_len = max_len - example.shape[-1]
        if mono_channel:
            batch.append(
                np.pad(example, [(0, 0), (0, pad_len)],
                       mode='constant',
                       constant_values=pad_value))
        else:
            batch.append(
                np.pad(example, [(0, 0), (0, 0), (0, pad_len)],
                       mode='constant',
                       constant_values=pad_value))  # what about PCM, no

    return np.array(batch, dtype=dtype)
