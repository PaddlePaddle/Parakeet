import paddle
paddle.set_default_dtype("float64")
paddle.disable_static(paddle.CPUPlace())
import unittest
import numpy as np

from parakeet.modules import conv

class TestConv1dCell(unittest.TestCase):
    def setUp(self):
        self.net = conv.Conv1dCell(4, 6, 5, dilation=2)
    
    def forward_incremental(self, x):
        outs = []
        self.net.start_sequence()
        with paddle.no_grad():
            for i in range(x.shape[-1]):
                xt = x[:, :, i]
                yt = self.net.add_input(xt)
                outs.append(yt)
            y2 = paddle.stack(outs, axis=-1)
        return y2
            
    def test_equality(self):
        x = paddle.randn([2, 4, 16])
        y1 = self.net(x)
        
        self.net.eval()
        y2 = self.forward_incremental(x)

        np.testing.assert_allclose(y2.numpy(), y1.numpy())
        
        
class TestConv1dBatchNorm(unittest.TestCase):
    def __init__(self, methodName="runTest", causal=False):
        super(TestConv1dBatchNorm, self).__init__(methodName)
        self.causal = causal
        
    def setUp(self):
        k = 5
        paddding = (k - 1, 0) if self.causal else ((k-1) // 2, k //2)
        self.net = conv.Conv1dBatchNorm(4, 6, (k,), 1, padding=paddding)

    def test_input_output(self):
        x = paddle.randn([4, 4, 16])
        out = self.net(x)
        out_np = out.numpy()
        self.assertTupleEqual(out_np.shape, (4, 6, 16))
        
    def runTest(self):
        self.test_input_output()


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    suite.addTest(TestConv1dBatchNorm("runTest", True))
    suite.addTest(TestConv1dBatchNorm("runTest", False))
    
    suite.addTest(TestConv1dCell("test_equality"))

    return suite