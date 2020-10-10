import unittest
import paddle
paddle.set_default_dtype("float64")
paddle.disable_static(paddle.CPUPlace())
from parakeet.modules import cbhg

class TestConv1dBatchNorm(unittest.TestCase):
    def __init__(self, methodName="runTest", causal=False):
        super(TestConv1dBatchNorm, self).__init__(methodName)
        self.causal = causal
        
    def setUp(self):
        k = 5
        paddding = (k - 1, 0) if self.causal else ((k-1) // 2, k //2)
        self.net = cbhg.Conv1dBatchNorm(4, 6, (k,), 1, padding=paddding)

    def test_input_output(self):
        x = paddle.randn([4, 4, 16])
        out = self.net(x)
        out_np = out.numpy()
        self.assertTupleEqual(out_np.shape, (4, 6, 16))
    
    def runTest(self):
        self.test_input_output()


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
    suite.addTest(TestConv1dBatchNorm("runTest", True))
    suite.addTest(TestConv1dBatchNorm("runTest", False))
    
    suite.addTest(TestHighway("test_io"))
    suite.addTest(TestCBHG("test_io"))
    return suite
