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
        