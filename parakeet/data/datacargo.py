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
from .sampler import SequentialSampler, RandomSampler, BatchSampler


class DataCargo(object):
    def __init__(self,
                 dataset,
                 batch_fn=None,
                 batch_size=1,
                 sampler=None,
                 shuffle=False,
                 batch_sampler=None,
                 drop_last=False):
        """An Iterable object of batches. It requires a dataset, a batch function and a sampler. The sampler yields the example ids, then the corresponding examples in the dataset are collected and transformed into a batch with the batch function.

        Args:
            dataset (Dataset): the dataset used to build a data cargo.
            batch_fn (callable, optional): a callable that takes a list of examples of `dataset` and return a batch, it can be None if the dataset has a `_batch_examples` method which satisfy the requirement. Defaults to None.
            batch_size (int, optional): number of examples in a batch. Defaults to 1.
            sampler (Sampler, optional): an iterable of example ids(intergers), the example ids are used to pick examples. Defaults to None.
            shuffle (bool, optional): when sampler is not provided, shuffle = True creates a RandomSampler and shuffle=False creates a SequentialSampler internally. Defaults to False.
            batch_sampler (BatchSampler, optional): an iterable of lists of example ids(intergers), the list is used to pick examples, `batch_sampler` option is mutually exclusive with `batch_size`, `shuffle`, `sampler`, and `drop_last`. Defaults to None.
            drop_last (bool, optional): whether to drop the last minibatch. Defaults to False.
        """
        self.dataset = dataset
        self.batch_fn = batch_fn or self.dataset._batch_examples

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            batch_size = None
            drop_last = False
            shuffle = False
        elif batch_size is None:
            raise ValueError(
                'batch sampler is none. then batch size must not be none.')
        elif sampler is None:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        else:
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler

        self.batch_sampler = batch_sampler

    def __iter__(self):
        return DataIterator(self)

    def __call__(self):
        # protocol for paddle's DataLoader
        return DataIterator(self)

    @property
    def _auto_collation(self):
        # use auto batching
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self):
        return len(self._index_sampler)


class DataIterator(object):
    def __init__(self, loader):
        """Iterator object of DataCargo.

        Args:
            loader (DataCargo): the data cargo to iterate.
        """
        self.loader = loader
        self._dataset = loader.dataset

        self._batch_fn = loader.batch_fn
        self._index_sampler = loader._index_sampler
        self._sampler_iter = iter(self._index_sampler)

    def __iter__(self):
        return self

    def __next__(self):
        # TODO(chenfeiyu): use dynamic batch size
        index = self._next_index()
        minibatch = [self._dataset[i] for i in index]
        minibatch = self._batch_fn(minibatch)  # list[Example] -> Batch
        return minibatch

    next = __next__  # Python 2 compatibility

    def _next_index(self):
        if six.PY3:
            return next(self._sampler_iter)
        else:
            # six.PY2
            return self._sampler_iter.next()

    def __len__(self):
        return len(self._index_sampler)
