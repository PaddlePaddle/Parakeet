import numpy as np
import unittest
import paddle
paddle.set_default_dtype("float64")
paddle.disable_static(paddle.CPUPlace())

from parakeet.models import deepvoice3 as dv3

class TestConvBlock(unittest.TestCase):
    def test_io_causal(self):
        net = dv3.ConvBlock(6, 5, True, True, 8, 0.9)
        x = paddle.randn([4, 32, 6])
        condition = paddle.randn([4, 8])
        # TODO(chenfeiyu): to report an issue on default data type
        padding = paddle.zeros([4, 4, 6], dtype=x.dtype)
        y = net.forward(x, condition, padding)
        self.assertTupleEqual(y.numpy().shape, (4, 32, 6))
        
    def test_io_non_causal(self):
        net = dv3.ConvBlock(6, 5, False, True, 8, 0.9)
        x = paddle.randn([4, 32, 6])
        condition = paddle.randn([4, 8])
        y = net.forward(x, condition)
        self.assertTupleEqual(y.numpy().shape, (4, 32, 6))
        
        
class TestAffineBlock1(unittest.TestCase):
    def test_io(self):
        net = dv3.AffineBlock1(6, 16, True, 8)
        x = paddle.randn([4, 32, 6])
        condition = paddle.randn([4, 8])
        y = net(x, condition)
        self.assertTupleEqual(y.numpy().shape, (4, 32, 16))
        

class TestAffineBlock2(unittest.TestCase):
    def test_io(self):
        net = dv3.AffineBlock2(6, 16, True, 8)
        x = paddle.randn([4, 32, 6])
        condition = paddle.randn([4, 8])
        y = net(x, condition)
        self.assertTupleEqual(y.numpy().shape, (4, 32, 16))
        

class TestEncoder(unittest.TestCase):
    def test_io(self):
        net = dv3.Encoder(5, 8, 16, 5, True, 6)
        x = paddle.randn([4, 32, 8])
        condition = paddle.randn([4, 6])
        keys, values = net(x, condition)
        self.assertTupleEqual(keys.numpy().shape, (4, 32, 8))
        self.assertTupleEqual(values.numpy().shape, (4, 32, 8))
        
        
class TestAttentionBlock(unittest.TestCase):
    def test_io(self):
        net = dv3.AttentionBlock(16, 6, has_bias=True, bias_dim=8)
        q = paddle.randn([4, 32, 6])
        k = paddle.randn([4, 24, 6])
        v = paddle.randn([4, 24, 6])
        lengths = paddle.to_tensor([24, 20, 19, 23], dtype="int64")
        condition = paddle.randn([4, 8])
        context_vector, attention_weight = net(q, k, v, lengths, condition, 0)
        self.assertTupleEqual(context_vector.numpy().shape, (4, 32, 6))
        self.assertTupleEqual(attention_weight.numpy().shape, (4, 32, 24))
        
    def test_io_with_previous_attn(self):
        net = dv3.AttentionBlock(16, 6, has_bias=True, bias_dim=8)
        q = paddle.randn([4, 32, 6])
        k = paddle.randn([4, 24, 6])
        v = paddle.randn([4, 24, 6])
        lengths = paddle.to_tensor([24, 20, 19, 23], dtype="int64")
        condition = paddle.randn([4, 8])
        prev_attn_weight = paddle.randn([4, 32, 16])
        
        context_vector, attention_weight = net(
            q, k, v, lengths, condition, 0, 
            force_monotonic=True, prev_coeffs=prev_attn_weight, window=(0, 4))
        self.assertTupleEqual(context_vector.numpy().shape, (4, 32, 6))
        self.assertTupleEqual(attention_weight.numpy().shape, (4, 32, 24))
        
        
class TestDecoder(unittest.TestCase):
    def test_io(self):
        net = dv3.Decoder(8, 4, [4, 12], 5, 3, 16, 1.0, 1.45, True, 6)
        x = paddle.randn([4, 32, 8])
        k = paddle.randn([4, 24, 12]) # prenet's last size should equals k's feature size
        v = paddle.randn([4, 24, 12])
        lengths = paddle.to_tensor([24, 18, 19, 22])
        condition = paddle.randn([4, 6])
        decoded, hidden, attentions, final_state = net(x, k, v, lengths, 0, condition)
        self.assertTupleEqual(decoded.numpy().shape, (4, 32, 4 * 8))
        self.assertTupleEqual(hidden.numpy().shape, (4, 32, 12))
        self.assertEqual(len(attentions), 5)
        self.assertTupleEqual(attentions[0].numpy().shape, (4, 32, 24))
        self.assertEqual(len(final_state), 5)
        self.assertTupleEqual(final_state[0].numpy().shape, (4, 2, 12))
        
        
class TestPostNet(unittest.TestCase):
    def test_io(self):
        net = dv3.PostNet(3, 8, 16, 3, 12, 4, True, 6)
        x = paddle.randn([4, 32, 8])
        condition = paddle.randn([4, 6])
        y = net(x, condition)
        self.assertTupleEqual(y.numpy().shape, (4, 32 * 4, 12))
        
