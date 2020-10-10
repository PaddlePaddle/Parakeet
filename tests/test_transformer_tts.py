import unittest
import numpy as np
import paddle
paddle.set_default_dtype("float64")
paddle.disable_static(paddle.CPUPlace())

from parakeet.models import transformer_tts as tts

class TestMultiheadAttention(unittest.TestCase):
    def test_io_same_qk(self):
        net = tts.MultiheadAttention(64, 8)
        q = paddle.randn([4, 12, 64])
        mask = paddle.fluid.layers.sequence_mask(
            paddle.to_tensor([12, 10, 8, 9]), dtype=q.dtype)
        mask = paddle.unsqueeze(mask, 1) # unsqueeze for time_steps_q
        context_vector, attention_weights = net(q, q, q, mask, drop_n_heads=2)
        self.assertTupleEqual(context_vector.numpy().shape, (4, 12, 64))
        self.assertTupleEqual(attention_weights.numpy().shape, (4, 8, 12, 12))
    
    def test_io(self):
        net = tts.MultiheadAttention(64, 8, k_dim=12, v_dim=6)
        q = paddle.randn([4, 12, 64])
        mask = paddle.fluid.layers.sequence_mask(
            paddle.to_tensor([12, 10, 8, 9]), dtype=q.dtype)
        mask = paddle.unsqueeze(mask, 1) # unsqueeze for time_steps_q
        context_vector, attention_weights = net(q, q, q, mask, drop_n_heads=2)
        self.assertTupleEqual(context_vector.numpy().shape, (4, 12, 64))
        self.assertTupleEqual(attention_weights.numpy().shape, (4, 8, 12, 12))
        
        
class TestTransformerEncoderLayer(unittest.TestCase):
    def test_io(self):
        net = tts.TransformerEncoderLayer(64, 8, 128)
        x = paddle.randn([4, 12, 64])
        mask = paddle.fluid.layers.sequence_mask(
            paddle.to_tensor([12, 10, 8, 9]), dtype=x.dtype)
        context_vector, attention_weights = net(x, mask)
        self.assertTupleEqual(context_vector.numpy().shape, (4, 12, 64))
        self.assertTupleEqual(attention_weights.numpy().shape, (4, 8, 12, 12))
        
        
class TestTransformerDecoderLayer(unittest.TestCase):
    def test_io(self):
        net = tts.TransformerDecoderLayer(64, 8, 128, 0.5)
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