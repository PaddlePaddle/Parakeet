import unittest
import paddle
paddle.set_default_dtype("float64")
paddle.disable_static(paddle.CPUPlace())
from parakeet.modules import cbhg


class TestHighway(unittest.TestCase):
    def test_io(self):
        net = cbhg.Highway(4)
        x = paddle.randn([2, 12, 4])
        y = net(x)
        self.assertTupleEqual(y.numpy().shape, (2, 12, 4))


class TestCBHG(unittest.TestCase):
    def __init__(self, methodName="runTest", ):
        super(TestCBHG, self).__init__(methodName)
    
    def test_io(self):
        self.net = cbhg.CBHG(64, 32, 16, 
                             projection_channels=[64, 128], 
                             num_highways=4, highway_features=128, 
                             gru_features=64)
        x = paddle.randn([4, 64, 32])
        y = self.net(x)
        self.assertTupleEqual(y.numpy().shape, (4, 32, 128))

def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    
    suite.addTest(TestHighway("test_io"))
    suite.addTest(TestCBHG("test_io"))
    return suite
