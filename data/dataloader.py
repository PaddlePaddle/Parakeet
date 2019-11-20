from sampler import SequentialSampler, RandomSampler, BatchSampler

class DataLoader(object):
    def __init__(self, dataset, batch_size=1, collate_fn = lambda x: x, 
                 sampler=None, shuffle=False, batch_sampler=None, drop_last=False):
        self.dataset = dataset
        self.collate_fn = collate_fn
        
        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if shuffle or drop_last:
                raise ValueError('batch_size=None option disables auto-batching '
                                 'and is mutually exclusive with '
                                 'shuffle, and drop_last')
        
        if sampler is None:  # give default samplers
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
    
    def __iter__(self):
        return DataIterator(self)
    
    @property
    def _auto_collation(self):
        # we will auto batching
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # (see _utils/fetch.py) to read data at each time. This would be
        # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
        # We can't change `.sampler` and `.batch_sampler` attributes for BC
        # reasons.
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self):
        return len(self._index_sampler)  # with iterable-style dataset, this will error
    
class DataIterator(object):
    def __init__(self, loader):
        self.loader = loader
        self._dataset = loader.dataset
        
        self._index_sampler = loader._index_sampler
        self._sampler_iter = iter(self._index_sampler)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        index = self._next_index()  # may raise StopIteration, TODO(chenfeiyu): use dynamic batch size
        minibatch = [self._dataset[i] for i in index] # we can abstract it, too to use dynamic batch size
        minibatch = self.loader.collate_fn(minibatch) # list[Example] -> Batch
        return minibatch
    
    def _next_index(self):
        return next(self._sampler_iter)
    
    def __len__(self):
        return len(self._index_sampler)
