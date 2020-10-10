import unittest
import paddle
from paddle import nn
paddle.disable_static(paddle.CPUPlace())
paddle.set_default_dtype("float64")

from parakeet.modules import connections as conn

class TestPreLayerNormWrapper(unittest.TestCase):
    def test_io(self):
        net = nn.Linear(8, 8)
        net = conn.PreLayerNormWrapper(net, 8)
        x = paddle.randn([4, 8])
        y = net(x)
        self.assertTupleEqual(x.numpy().shape, y.numpy().shape)
        

class TestPostLayerNormWrapper(unittest.TestCase):
    def test_io(self):
        net = nn.Linear(8, 8)
        net = conn.PostLayerNormWrapper(net, 8)
        x = paddle.randn([4, 8])
        y = net(x)
        self.assertTupleEqual(x.numpy().shape, y.numpy().shape)
        
        
class TestResidualWrapper(unittest.TestCase):
    def test_io(self):
        net = nn.Linear(8, 8)
        net = conn.ResidualWrapper(net)
        x = paddle.randn([4, 8])
        y = net(x)
        self.assertTupleEqual(x.numpy().shape, y.numpy().shape)