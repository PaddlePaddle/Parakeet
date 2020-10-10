import unittest
import numpy as np
import paddle
from paddle import io
from parakeet import data

class MyDataset(io.Dataset):
    def __init__(self, size):
        self._data = np.random.randn(size, 6)
    
    def __getitem__(self, i):
        return self._data[i]
    
    def __len__(self):
        return self._data.shape[0]


class TestTransformDataset(unittest.TestCase):
    def test(self):
        dataset = MyDataset(20)
        dataset = data.TransformDataset(dataset, lambda x: np.abs(x))
        dataloader = io.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
        print("TransformDataset")
        for batch, in dataloader:
            print(type(batch), batch.dtype, batch.shape)


class TestChainDataset(unittest.TestCase):
    def test(self):
        dataset1 = MyDataset(20)
        dataset2 = MyDataset(40)
        dataset = data.ChainDataset(dataset1, dataset2)
        dataloader = io.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
        print("ChainDataset")
        for batch, in dataloader:
            print(type(batch), batch.dtype, batch.shape)


class TestTupleDataset(unittest.TestCase):
    def test(self):
        dataset1 = MyDataset(20)
        dataset2 = MyDataset(20)
        dataset = data.TupleDataset(dataset1, dataset2)
        dataloader = io.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
        print("TupleDataset")
        for field1, field2 in dataloader:
            print(type(field1), field1.dtype, field1.shape)
            print(type(field2), field2.dtype, field2.shape)


class TestDictDataset(unittest.TestCase):
    def test(self):
        dataset1 = MyDataset(20)
        dataset2 = MyDataset(20)
        dataset = data.DictDataset(field1=dataset1, field2=dataset2)
        def collate_fn(examples):
            examples_tuples = []
            for example in examples:
                examples_tuples.append(example.values())
            return paddle.fluid.dataloader.dataloader_iter.default_collate_fn(examples_tuples)
            
        dataloader = io.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=collate_fn)
        print("DictDataset")
        for field1, field2 in dataloader:
            print(type(field1), field1.dtype, field1.shape)
            print(type(field2), field2.dtype, field2.shape)


class TestSliceDataset(unittest.TestCase):
    def test(self):
        dataset = MyDataset(40)
        dataset = data.SliceDataset(dataset, 0, 20)
        dataloader = io.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
        print("SliceDataset")
        for batch, in dataloader:
            print(type(batch), batch.dtype, batch.shape)


class TestSplit(unittest.TestCase):
    def test(self):
        dataset = MyDataset(40)
        train, valid = data.split(dataset, 10)
        dataloader1 = io.DataLoader(train, batch_size=4, shuffle=True, num_workers=1)
        dataloader2 = io.DataLoader(valid, batch_size=4, shuffle=True, num_workers=1)
        print("First Dataset")
        for batch, in dataloader1:
            print(type(batch), batch.dtype, batch.shape)
            
        print("Second Dataset")
        for batch, in dataloader2:
            print(type(batch), batch.dtype, batch.shape)


class TestSubsetDataset(unittest.TestCase):
    def test(self):
        dataset = MyDataset(40)
        indices = np.random.choice(np.arange(40), [20], replace=False).tolist()
        dataset = data.SubsetDataset(dataset, indices)
        dataloader = io.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
        print("SubsetDataset")
        for batch, in dataloader:
            print(type(batch), batch.dtype, batch.shape)


class TestFilterDataset(unittest.TestCase):
    def test(self):
        dataset = MyDataset(40)
        dataset = data.FilterDataset(dataset, lambda x: np.mean(x)> 0.3)
        dataloader = io.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
        print("FilterDataset")
        for batch, in dataloader:
            print(type(batch), batch.dtype, batch.shape)


class TestCacheDataset(unittest.TestCase):
    def test(self):
        dataset = MyDataset(40)
        dataset = data.CacheDataset(dataset)
        dataloader = io.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
        print("CacheDataset")
        for batch, in dataloader:
            print(type(batch), batch.dtype, batch.shape)
