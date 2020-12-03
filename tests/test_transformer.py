import unittest
import numpy as np
import paddle
paddle.set_default_dtype("float64")
paddle.disable_static(paddle.CPUPlace())

from parakeet.modules import transformer

class TestPositionwiseFFN(unittest.TestCase):
    def test_io(self):
        net = transformer.PositionwiseFFN(8, 12)
        x = paddle.randn([2, 3, 4, 8])
        y = net(x)
        self.assertTupleEqual(y.numpy().shape, (2, 3, 4, 8))


class TestTransformerEncoderLayer(unittest.TestCase):
    def test_io(self):
        net = transformer.TransformerEncoderLayer(64, 8, 128, 0.5)
        x = paddle.randn([4, 12, 64])
        lengths = paddle.to_tensor([12, 8, 9, 10])
        mask = paddle.fluid.layers.sequence_mask(lengths, dtype=x.dtype)
        y, attn_weights = net(x, mask)
        
        self.assertTupleEqual(y.numpy().shape, (4, 12, 64))
        self.assertTupleEqual(attn_weights.numpy().shape, (4, 8, 12, 12))


class TestTransformerDecoderLayer(unittest.TestCase):
    def test_io(self):
        net = transformer.TransformerDecoderLayer(64, 8, 128, 0.5)
        q = paddle.randn([4, 32, 64])
        k = paddle.randn([4, 24, 64])
        v = paddle.randn([4, 24, 64])
        enc_lengths = paddle.to_tensor([24, 18, 20, 22])
        dec_lengths = paddle.to_tensor([32, 28, 30, 31])
        enc_mask = paddle.fluid.layers.sequence_mask(enc_lengths, dtype=k.dtype)
        dec_mask = paddle.fluid.layers.sequence_mask(dec_lengths, dtype=q.dtype)
        y, self_attn_weights, cross_attn_weights = net(q, k, v, enc_mask, dec_mask)
        
        self.assertTupleEqual(y.numpy().shape, (4, 32, 64))
        self.assertTupleEqual(self_attn_weights.numpy().shape, (4, 8, 32, 32))
        self.assertTupleEqual(cross_attn_weights.numpy().shape, (4, 8, 32, 24))