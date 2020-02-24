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
        return DataIterator(self)

    @property
    def _auto_collation(self):
        # we will auto batching
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
        self.loader = loader
        self._dataset = loader.dataset

        self._batch_fn = loader.batch_fn
        self._index_sampler = loader._index_sampler
        self._sampler_iter = iter(self._index_sampler)

    def __iter__(self):
        return self

    def __next__(self):

        index = self._next_index(
        )  # may raise StopIteration, TODO(chenfeiyu): use dynamic batch size
        minibatch = [self._dataset[i] for i in index
                     ]  # we can abstract it, too to use dynamic batch size
        minibatch = self._batch_fn(minibatch)  # list[Example] -> Batch
        return minibatch

    def _next_index(self):
        if six.PY3:
            return next(self._sampler_iter)
        else:
            # six.PY2
            return self._sampler_iter.next()

    def __len__(self):
        return len(self._index_sampler)
