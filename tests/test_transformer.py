import unittest
import numpy as np
import paddle
paddle.set_default_dtype("float64")
paddle.disable_static(paddle.CPUPlace())

from parakeet.modules import transformer

def sequence_mask(lengths, max_length=None, dtype="bool"):
    max_length = max_length or np.max(lengths)
    ids = np.arange(max_length)
    return (ids < np.expand_dims(lengths, -1)).astype(dtype)

def future_mask(lengths, max_length=None, dtype="bool"):
    max_length = max_length or np.max(lengths)
    return np.tril(np.tril(np.ones(max_length)))

class TestPositionwiseFFN(unittest.TestCase):
    def test_io(self):
        net = transformer.PositionwiseFFN(8, 12)
        x = paddle.randn([2, 3, 4, 8])
        y = net(x)
        self.assertTupleEqual(y.numpy().shape, (2, 3, 4, 8))


class TestCombineMask(unittest.TestCase):
    def test_equality(self):
        lengths = np.array([12, 8, 9, 10])
        padding_mask = sequence_mask(lengths, dtype="float64")
        no_future_mask = future_mask(lengths, dtype="float64")
        combined_mask1 = np.expand_dims(padding_mask, 1) * no_future_mask
        
        combined_mask2 = transformer.combine_mask(
            paddle.to_tensor(padding_mask), paddle.to_tensor(no_future_mask)
        )
        np.testing.assert_allclose(combined_mask2.numpy(), combined_mask1)


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