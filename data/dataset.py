class Dataset(object):
    def __init__(self, lazy=True, stream=False):
        # note that lazy and stream means two different things in our glossary
        # lazy means to place preprocessing in __getitem__
        # stram means the data source is itself a stream
        self.lazy = lazy
        self.stream = stream
    
    def _load_metadata(self):
        raise NotImplementedError
    
    def _get_example(self):
        """return a Record"""
        raise NotImplementedError
    
    def _prepare_metadata(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __iter__(self):
        raise NotImplementedError

