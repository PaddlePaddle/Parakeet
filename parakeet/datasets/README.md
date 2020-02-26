# The Design of Dataset in Parakeet

## data & metadata
A Dataset in Parakeet is basically a list of Records (or examples, instances if you prefer this glossary.) By being a list, we mean it can be indexed by `__getitem__`, and we can get the size of the dataset by `__len__`.

This might mean we should have load the whole dataset before hand. But in practice, we do not do this due to time, computation and memory of storage limits. We actually load some metadata instead, which gives us the size of the dataset, and metadata of each record. In this case, the metadata itself is a small dataset which helps us to load a larger dataset. We made `_load_metadata` a method for all datasets.

In most cases, metadata is provided with the data. So we can load it trivially. But in other cases, we need to scan the whole dataset to get metadata. For example, the length of the the sentences, the vocabuary or the statistics of the dataset, etc. In these cases, we'd betetr save the metadata, so we do not need to generate them again and again. When implementing a dataset, we do these work in `_prepare_metadata`.

In our initial cases, record is implemented as a tuple for simplicity. Actually, it can be implemented as a dict or namespace.

## preprocessing & batching
One of the reasons we choose to load data lazily (only load metadata before hand, and load data only when needed) is computation overhead. For large dataset with complicated preprocessing, it may take several days to preprocess them. So we choose to preprocess it lazily. In practice, we implement preprocessing in `_get_example` which is called by `__getitem__`. This method preprocess only one record.

For deep learning practice, we typically batch examples. So the dataset should comes with a method to batch examples. Assuming the record is implemented as a tuple with several items. When an item is represented as a fix-sized array, to batch them is trivial, just `np.stack` suffices. But for array with dynamic size, padding is needed. We decide to implement a batching method for each item. Then batching a record can be implemented by these methods. For a dataset, a `_batch_examples` should be implemented. But in most cases, you can choose one from `batching.py`.

That is it!
