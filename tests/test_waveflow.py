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

    def test_add_input(self):
        net = waveflow.ResidualBlock(4, 6, (3, 3), (2, 2))
        net.eval()
        net.start_sequence()

        x_row = paddle.randn([4, 4, 1, 32])
        condition_row = paddle.randn([4, 6, 1, 32])

        res, skip = net.add_input(x_row, condition_row)
        self.assertTupleEqual(res.numpy().shape, (4, 4, 1, 32))
        self.assertTupleEqual(skip.numpy().shape, (4, 4, 1, 32))
        
        
class TestResidualNet(unittest.TestCase):
    def test_io(self):
        net = waveflow.ResidualNet(8, 6, 8, (3, 3), [1, 1, 1, 1, 1, 1, 1, 1])
        x = paddle.randn([4, 6, 8, 32])
        condition = paddle.randn([4, 8, 8, 32])
        y = net(x, condition)
        self.assertTupleEqual(y.numpy().shape, (4, 6, 8, 32))

    def test_add_input(self):
        net = waveflow.ResidualNet(8, 6, 8, (3, 3), [1, 1, 1, 1, 1, 1, 1, 1])
        net.eval()
        net.start_sequence()

        x_row = paddle.randn([4, 6, 1, 32])
        condition_row = paddle.randn([4, 8, 1, 32])

        y_row = net.add_input(x_row, condition_row)
        self.assertTupleEqual(y_row.numpy().shape, (4, 6, 1, 32))
        
        
class TestFlow(unittest.TestCase):
    def test_io(self):
        net = waveflow.Flow(8, 16, 7, (3, 3), 8)

        x = paddle.randn([4, 1, 8, 32])
        condition = paddle.randn([4, 7, 8, 32])
        z, (logs, b) = net(x, condition)
        self.assertTupleEqual(z.numpy().shape, (4, 1, 8, 32))
        self.assertTupleEqual(logs.numpy().shape, (4, 1, 7, 32))
        self.assertTupleEqual(b.numpy().shape, (4, 1, 7, 32))
    
    def test_inverse_row(self):
        net = waveflow.Flow(8, 16, 7, (3, 3), 8)
        net.eval()
        net.start_sequence()

        x_row = paddle.randn([4, 1, 1, 32]) # last row
        condition_row = paddle.randn([4, 7, 1, 32])
        z_row = paddle.randn([4, 1, 1, 32])
        x_next_row, (logs, b) = net._inverse_row(z_row, x_row, condition_row)

        self.assertTupleEqual(x_next_row.numpy().shape, (4, 1, 1, 32))
        self.assertTupleEqual(logs.numpy().shape, (4, 1, 1, 32))
        self.assertTupleEqual(b.numpy().shape, (4, 1, 1, 32))

    def test_inverse(self):
        net = waveflow.Flow(8, 16, 7, (3, 3), 8)
        net.eval()
        net.start_sequence()

        z = paddle.randn([4, 1, 8, 32])
        condition = paddle.randn([4, 7, 8, 32])

        with paddle.no_grad():
            x, (logs, b) = net.inverse(z, condition)
        self.assertTupleEqual(x.numpy().shape, (4, 1, 8, 32))
        self.assertTupleEqual(logs.numpy().shape, (4, 1, 7, 32))
        self.assertTupleEqual(b.numpy().shape, (4, 1, 7, 32))


class TestWaveFlow(unittest.TestCase):
    def test_io(self):
        x = paddle.randn([4, 32 * 8 ])
        condition = paddle.randn([4, 7, 32 * 8])
        net = waveflow.WaveFlow(2, 8, 8, 16, 7, (3, 3))
        z, logs_det_jacobian = net(x, condition)

        self.assertTupleEqual(z.numpy().shape, (4, 32 * 8))
        self.assertTupleEqual(logs_det_jacobian.numpy().shape, (1,))

    def test_inverse(self):
        z = paddle.randn([4, 32 * 8 ])
        condition = paddle.randn([4, 7, 32 * 8])

        net = waveflow.WaveFlow(2, 8, 8, 16, 7, (3, 3))
        net.eval()

        with paddle.no_grad():
            x = net.inverse(z, condition)
        self.assertTupleEqual(x.numpy().shape, (4, 32 * 8))
        

        