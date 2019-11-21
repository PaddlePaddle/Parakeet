class Dataset(object):
    def __init__(self):
        pass
    
    def _load_metadata(self):
        raise NotImplementedError
    
    def _get_example(self):
        """return a Record (or Example, Instance according to your glossary)"""
        raise NotImplementedError
    
    def _batch_examples(self, minibatch):
        """get a list of examples, return a batch, whose structure is the same as an example"""
        raise NotImplementedError
    
    def _prepare_metadata(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __iter__(self):
        raise NotImplementedError

