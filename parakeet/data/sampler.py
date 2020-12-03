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
At most cases, we have non-stream dataset, which means we can random access it with __getitem__, and we can get the length of the dataset with __len__.

This suffices for a sampler. We implemente sampler as iterable of valid indices. By valid, we mean 0 <= index < N, where N is the length of the dataset. We then collect several indices within a batch and use them to collect examples from the dataset with __getitem__. Then transform these examples into a batch.

So the sampler is only responsible for generating valid indices.
"""

import numpy as np
import random
import paddle
from paddle.io import Sampler


class PartialyRandomizedSimilarTimeLengthSampler(Sampler):
    """Partially randmoized sampler, implemented as a example sampler
    1. Sort by lengths
    2. Pick a small patch and randomize it
    3. Permutate mini-batchs
    """

    def __init__(self,
                 lengths,
                 batch_size=4,
                 batch_group_size=None,
                 permutate=True):
        """[summary]

        Args:
            lengths (List[int]): The length of the examples of the dataset. This is the key to be considered as 'time length'.
            batch_size (int, optional): batch size. Defaults to 4.
            batch_group_size (int, optional): the size of a small batch. Random shuffling is applied within such patches. If `batch_group_size` is not provided, it is set to min(batch_size * 32, len(self.lengths)). Batch_group_size should be perfectly divided by batch_size. Defaults to None.
            permutate (bool, optional): permutate batches. Defaults to True.
        """
        _lengths = np.array(
            lengths,
            dtype=np.int64)  # maybe better implement length as a sort key
        self.lengths = np.sort(_lengths)
        self.sorted_indices = np.argsort(_lengths)

        self.batch_size = batch_size
        if batch_group_size is None:
            batch_group_size = min(batch_size * 32, len(self.lengths))
            if batch_group_size % batch_size != 0:
                batch_group_size -= batch_group_size % batch_size

        self.batch_group_size = batch_group_size
        assert batch_group_size % batch_size == 0
        self.permutate = permutate

    def __iter__(self):
        indices = np.copy(self.sorted_indices)
        batch_group_size = self.batch_group_size
        s, e = 0, 0
        for i in range(len(indices) // batch_group_size):
            s = i * batch_group_size
            e = s + batch_group_size
            random.shuffle(indices[s:e])  # inplace

        # Permutate batches
        if self.permutate:
            perm = np.arange(len(indices[:e]) // self.batch_size)
            random.shuffle(perm)
            indices[:e] = indices[:e].reshape(
                -1, self.batch_size)[perm, :].reshape(-1)

        # Handle last elements
        s += batch_group_size
        #print(indices)
        if s < len(indices):
            random.shuffle(indices[s:])

        return iter(indices)

    def __len__(self):
        return len(self.sorted_indices)


class BucketSampler(Sampler):
    def __init__(self,
                 lengths,
                 batch_size=4,
                 batch_group_size=None,
                 permutate=True,
                 num_trainers=1,
                 rank=0):
        # maybe better implement length as a sort key
        _lengths = np.array(lengths, dtype=np.int64)
        self.lengths = np.sort(_lengths)
        self.sorted_indices = np.argsort(_lengths)
        self.num_trainers = num_trainers
        self.rank = rank

        self.dataset_size = len(_lengths)
        self.num_samples = int(np.ceil(self.dataset_size / num_trainers))
        self.total_size = self.num_samples * num_trainers
        assert self.total_size >= self.dataset_size

        self.batch_size = batch_size
        total_batch_size = num_trainers * batch_size
        self.total_batch_size = total_batch_size

        if batch_group_size is None:
            batch_group_size = min(total_batch_size * 32, len(self.lengths))
            if batch_group_size % total_batch_size != 0:
                batch_group_size -= batch_group_size % total_batch_size

        self.batch_group_size = batch_group_size
        assert batch_group_size % total_batch_size == 0
        self.permutate = permutate

    def __iter__(self):
        indices = self.sorted_indices

        # Append extra samples to make it evenly distributed on all trainers.
        num_extras = self.total_size - self.dataset_size
        extra_indices = np.random.choice(
            indices, size=(num_extras, ), replace=False)
        indices = np.concatenate((indices, extra_indices))
        assert len(indices) == self.total_size

        batch_group_size = self.batch_group_size
        s, e = 0, 0
        for i in range(len(indices) // batch_group_size):
            s = i * batch_group_size
            e = s + batch_group_size
            random.shuffle(indices[s:e])  # inplace

        # Permutate batches
        total_batch_size = self.total_batch_size
        if self.permutate:
            perm = np.arange(len(indices[:e]) // total_batch_size)
            random.shuffle(perm)
            indices[:e] = indices[:e].reshape(
                -1, total_batch_size)[perm, :].reshape(-1)

        # Handle last elements
        s += batch_group_size
        #print(indices)
        if s < len(indices):
            random.shuffle(indices[s:])

        # Subset samples for each trainer.
        indices = indices[self.rank:self.total_size:self.num_trainers]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return len(self.sorted_indices)


class WeightedRandomSampler(Sampler):
    """Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).
    Args:
        weights (List[float]): a sequence of weights, not necessary summing up to 1.
        num_samples (int): number of samples to draw.
        replacement (bool): whether samples are drawn with replacement. When replacement is False, num_samples should not be larger than len(weights).
    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [0, 0, 0, 1, 0]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """

    def __init__(self, weights, num_samples, replacement):
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(
                                 num_samples))
        self.weights = np.array(weights, dtype=np.float64)
        self.num_samples = num_samples
        self.replacement = replacement
        if replacement is False and num_samples > len(weights):
            raise ValueError(
                "when replacement is False, num_samples should not be"
                "larger that length of weight.")

    def __iter__(self):
        return iter(
            np.random.choice(
                len(self.weights),
                size=(self.num_samples, ),
                replace=self.replacement,
                p=self.weights).tolist())

    def __len__(self):
        return self.num_samples
