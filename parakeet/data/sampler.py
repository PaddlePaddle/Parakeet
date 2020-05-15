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


class Sampler(object):
    def __iter__(self):
        # return a iterator of indices
        # or a iterator of list[int], for BatchSampler
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, data_source):
        """Sequential sampler, the simplest sampler that samples indices from 0 to N - 1, where N is the dataset is length.

        Args:
            data_source (DatasetMixin): the dataset. This is used to get the dataset's length.
        """
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None):
        """Random sampler.

        Args:
            data_source (DatasetMixin): the dataset. This is used to get the dataset's length.
            replacement (bool, optional): whether replacement is enabled in sampling. When `replacement` is True, `num_samples` must be provided. Defaults to False.
            num_samples (int, optional): numbers of indices to draw. This option should only be provided when replacement is True. Defaults to None.
        """
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError(
                "With replacement=False, num_samples should not be specified, "
                "since a random permutation will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(
                                 self.num_samples))

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(
                np.random.randint(
                    0, n, size=(self.num_samples, ), dtype=np.int64).tolist())
        return iter(np.random.permutation(n).tolist())

    def __len__(self):
        return self.num_samples


class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        """
        Args:
            indices (List[int]): indices to sample from.
        """
        self.indices = indices

    def __iter__(self):
        return (self.indices[i]
                for i in np.random.permutation(len(self.indices)))

    def __len__(self):
        return len(self.indices)


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


class DistributedSampler(Sampler):
    def __init__(self, dataset_size, num_trainers, rank, shuffle=True):
        """Sampler used for data parallel training. Indices are divided into num_trainers parts. Each trainer gets a subset and iter that subset. If the dataset has 16 examples, and there are 4 trainers. 

        Trainer 0 gets [0, 4, 8, 12];
        Trainer 1 gets [1, 5, 9, 13];
        Trainer 2 gets [2, 6, 10, 14];
        trainer 3 gets [3, 7, 11, 15].

        It ensures that trainer get different parts of the dataset. If dataset's length cannot be perfectly devidef by num_trainers, some examples appended to the dataset, to ensures that every trainer gets the same amounts of examples.

        Args:
            dataset_size (int): the length of the dataset.
            num_trainers (int): number of trainers(training processes).
            rank (int): local rank of the trainer.
            shuffle (bool, optional): whether to shuffle the indices before iteration. Defaults to True.
        """
        self.dataset_size = dataset_size
        self.num_trainers = num_trainers
        self.rank = rank
        self.num_samples = int(np.ceil(dataset_size / num_trainers))
        self.total_size = self.num_samples * num_trainers
        assert self.total_size >= self.dataset_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(self.dataset_size))
        if self.shuffle:
            random.shuffle(indices)

        # Append extra samples to make it evenly distributed on all trainers.
        indices += indices[:(self.total_size - self.dataset_size)]
        assert len(indices) == self.total_size

        # Subset samples for each trainer.
        indices = indices[self.rank:self.total_size:self.num_trainers]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


class BatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices."""

    def __init__(self, sampler, batch_size, drop_last):
        """
        Args:
            sampler (Sampler): Base sampler.
            batch_size (int): Size of mini-batch.
            drop_last (bool): If True, the sampler will drop the last batch if its size is less than batch_size.
        Example:
            >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
            >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        """
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "Sampler, but got sampler={}".format(sampler))
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
