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

import six
import numpy as np


class DatasetMixin(object):
    """standard indexing interface for dataset."""

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [
                self.get_example(i) for i in six.moves.range(start, stop, step)
            ]
        elif isinstance(index, (list, np.ndarray)):
            return [self.get_example(i) for i in index]
        else:
            # assumes it an integer
            return self.get_example(index)

    def get_example(self, i):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_example(i)


class TransformDataset(DatasetMixin):
    """Transform a dataset to another with a transform."""

    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def get_example(self, i):
        # CAUTION: only int is supported?
        # CAUTION: dataset support support __getitem__ and __len__
        in_data = self._dataset[i]
        return self._transform(in_data)


class CacheDataset(DatasetMixin):
    def __init__(self, dataset):
        self._dataset = dataset
        self._cache = dict()

    def __len__(self):
        return len(self._dataset)

    def get_example(self, i):
        if not i in self._cache:
            self._cache[i] = self._dataset[i]
        return self._cache[i]


class TupleDataset(object):
    def __init__(self, *datasets):
        if not datasets:
            raise ValueError("no datasets are given")
        length = len(datasets[0])
        for i, dataset in enumerate(datasets):
            if len(datasets) != length:
                raise ValueError(
                    "all the datasets should have the same length."
                    "dataset {} has a different length".format(i))
        self._datasets = datasets
        self._length = length

    def __getitem__(self, index):
        # SOA
        batches = [dataset[index] for dataset in self._datasets]
        if isinstance(index, slice):
            length = len(batches[0])
            # AOS
            return [
                tuple([batch[i] for batch in batches])
                for i in six.moves.range(length)
            ]
        else:
            return tuple(batches)

    def __len__(self):
        return self._length


class DictDataset(object):
    def __init__(self, **datasets):
        if not datasets:
            raise ValueError("no datasets are given")
        length = None
        for key, dataset in six.iteritems(datasets):
            if length is None:
                length = len(dataset)
            elif len(datasets) != length:
                raise ValueError(
                    "all the datasets should have the same length."
                    "dataset {} has a different length".format(key))
        self._datasets = datasets
        self._length = length

    def __getitem__(self, index):
        batches = {
            key: dataset[index]
            for key, dataset in six.iteritems(self._datasets)
        }
        if isinstance(index, slice):
            length = len(six.next(six.itervalues(batches)))
            return [{key: batch[i]
                     for key, batch in six.iteritems(batches)}
                    for i in six.moves.range(length)]
        else:
            return batches


class SliceDataset(DatasetMixin):
    def __init__(self, dataset, start, finish, order=None):
        if start < 0 or finish > len(dataset):
            raise ValueError("subset overruns the dataset.")
        self._dataset = dataset
        self._start = start
        self._finish = finish
        self._size = finish - start

        if order is not None and len(order) != len(dataset):
            raise ValueError(
                "order should have the same length as the dataset"
                "len(order) = {} which does not euqals len(dataset) = {} ".
                format(len(order), len(dataset)))
        self._order = order

    def __len__(self):
        return self._size

    def get_example(self, i):
        if i >= 0:
            if i >= self._size:
                raise IndexError('dataset index out of range')
            index = self._start + i
        else:
            if i < -self._size:
                raise IndexError('dataset index out of range')
            index = self._finish + i

        if self._order is not None:
            index = self._order[index]
        return self._dataset[index]


class SubsetDataset(DatasetMixin):
    def __init__(self, dataset, indices):
        self._dataset = dataset
        if len(indices) > len(dataset):
            raise ValueError("subset's size larger that dataset's size!")
        self._indices = indices
        self._size = len(indices)

    def __len__(self):
        return self._size

    def get_example(self, i):
        index = self._indices[i]
        return self._dataset[index]


class FilterDataset(DatasetMixin):
    def __init__(self, dataset, filter_fn):
        self._dataset = dataset
        self._indices = [
            i for i in range(len(dataset)) if filter_fn(dataset[i])
        ]
        self._size = len(self._indices)

    def __len__(self):
        return self._size

    def get_example(self, i):
        index = self._indices[i]
        return self._dataset[index]


class ChainDataset(DatasetMixin):
    def __init__(self, *datasets):
        self._datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def get_example(self, i):
        if i < 0:
            raise IndexError("ChainDataset doesnot support negative indexing.")

        for dataset in self._datasets:
            if i < len(dataset):
                return dataset[i]
            i -= len(dataset)

        raise IndexError("dataset index out of range")
