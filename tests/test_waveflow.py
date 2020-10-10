import numpy as np
import unittest

import paddle
paddle.set_default_dtype("float64")
paddle.disable_static(paddle.CPUPlace())

from parakeet.models import waveflow

class TestFold(unittest.TestCase):
    def test_audio(self):
        x = paddle.randn([4, 32 * 8])
        y = waveflow.fold(x, 8)
        self.assertTupleEqual(y.numpy().shape, (4, 32, 8))
    
    def test_spec(self):
        x = paddle.randn([4, 80, 32 * 8])
        y = waveflow.fold(x, 8)
        self.assertTupleEqual(y.numpy().shape, (4, 80, 32, 8))


class TestUpsampleNet(unittest.TestCase):
    def test_io(self):
        net = waveflow.UpsampleNet([2, 2])
        x = paddle.randn([4, 8, 6])
        y = net(x)
        self.assertTupleEqual(y.numpy().shape, (4, 8, 2 * 2 * 6))
        

class TestResidualBlock(unittest.TestCase):
    def test_io(self):
        net = waveflow.ResidualBlock(4, 6, (3, 3), (2, 2))
        x = paddle.randn([4, 4, 16, 32])
        condition = paddle.randn([4, 6, 16, 32])
        res, skip = net(x, condition)
        self.assertTupleEqual(res.numpy().shape, (4, 4, 16, 32))
        self.assertTupleEqual(skip.numpy().shape, (4, 4, 16, 32))
        
        
class TestResidualNet(unittest.TestCase):
    def test_io(self):
        net = waveflow.ResidualNet(8, 6, 8, (3, 3), [1, 1, 1, 1, 1, 1, 1, 1])
        x = paddle.randn([4, 6, 8, 32])
        condition = paddle.randn([4, 8, 8, 32])
        y = net(x, condition)
        self.assertTupleEqual(y.numpy().shape, (4, 6, 8, 32))
        
        
class TestFlow(unittest.TestCase):
    def test_io(self):
        x = paddle.randn([4, 1, 8, 32])
        condition = paddle.randn([4, 7, 8, 32])
        net = waveflow.Flow(8, 16, 7, (3, 3), 8)
        y = net(x, condition)
        self.assertTupleEqual(y.numpy().shape, (4, 2, 8, 32))
        
        
class TestWaveflow(unittest.TestCase):
    def test_io(self):
        x = paddle.randn([4, 32 * 8 ])
        condition = paddle.randn([4, 7, 32 * 8])
        net = waveflow.WaveFlow(2, 8, 8, 16, 7, (3, 3))
        z, logs = net(x, condition)
        