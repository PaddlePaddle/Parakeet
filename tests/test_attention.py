import unittest
import numpy as np
import paddle
paddle.set_default_dtype("float64")
paddle.disable_static(paddle.CPUPlace())

from parakeet.modules import attention as attn

class TestScaledDotProductAttention(unittest.TestCase):
    def test_without_mask(self):
        x = paddle.randn([4, 16, 8])
        context_vector, attention_weights = attn.scaled_dot_product_attention(x, x, x)
        assert(list(context_vector.shape) == [4, 16, 8])
        assert(list(attention_weights.shape) == [4, 16, 16])
        
    def test_with_mask(self):
        x = paddle.randn([4, 16, 8])
        mask = paddle.fluid.layers.sequence_mask(
            paddle.to_tensor([16, 15, 13, 14]), dtype=x.dtype)
        mask = mask.unsqueeze(1) # unsqueeze for the decoder time steps
        context_vector, attention_weights = attn.scaled_dot_product_attention(x, x, x, mask)
        assert(list(context_vector.shape) == [4, 16, 8])
        assert(list(attention_weights.shape) == [4, 16, 16])
        
    def test_4d(self):
        x = paddle.randn([4, 6, 16, 8])
        context_vector, attention_weights = attn.scaled_dot_product_attention(x, x, x)
        assert(list(context_vector.shape) == [4, 6, 16, 8])
        assert(list(attention_weights.shape) == [4, 6, 16, 16])


class TestMonoheadAttention(unittest.TestCase):
    def test_io(self):
        net = attn.MonoheadAttention(6, 0.1)
        q = paddle.randn([4, 18, 6])
        k = paddle.randn([4, 12, 6])
        v = paddle.randn([4, 12, 6])
        mask = paddle.fluid.layers.sequence_mask(
            paddle.to_tensor([12, 10, 8, 9]), dtype=q.dtype)
        mask = paddle.unsqueeze(mask, 1) # unsqueeze for time_steps_q
        context_vector, attn_weights = net(q, k, v, mask)
        self.assertTupleEqual(context_vector.numpy().shape, (4, 18, 6))
        self.assertTupleEqual(attn_weights.numpy().shape, (4, 18, 12))


class TestDropHead(unittest.TestCase):
    def test_drop(self):
        x = paddle.randn([4, 6, 16, 8])
        out = attn.drop_head(x, 2, training=True)
        # drop 2 head from 6 at all positions
        np.testing.assert_allclose(np.sum(out.numpy() == 0., axis=1), 2)
    
    def test_drop_all(self):
        x = paddle.randn([4, 6, 16, 8])
        out = attn.drop_head(x, 6, training=True)
        np.testing.assert_allclose(np.sum(out.numpy()), 0)
    
    def test_eval(self):
        x = paddle.randn([4, 6, 16, 8])
        out = attn.drop_head(x, 6, training=False)
        self.assertIs(x, out)


class TestMultiheadAttention(unittest.TestCase):
    def __init__(self, methodName="test_io", same_qk=True):
        super(TestMultiheadAttention, self).__init__(methodName)
        self.same_qk = same_qk
    
    def setUp(self):
        if self.same_qk:
            net = attn.MultiheadAttention(64, 8, dropout=0.3)
        else:
            net = attn.MultiheadAttention(64, 8, k_dim=12, v_dim=6)
        self.net =net
            
    def test_io(self):
        q = paddle.randn([4, 12, 64])
        mask = paddle.fluid.layers.sequence_mask(
            paddle.to_tensor([12, 10, 8, 9]), dtype=q.dtype)
        mask = paddle.unsqueeze(mask, 1) # unsqueeze for time_steps_q
        context_vector, attention_weights = self.net(q, q, q, mask)
        self.assertTupleEqual(context_vector.numpy().shape, (4, 12, 64))
        self.assertTupleEqual(attention_weights.numpy().shape, (4, 8, 12, 12))


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    suite.addTest(TestScaledDotProductAttention("test_without_mask"))
    suite.addTest(TestScaledDotProductAttention("test_with_mask"))
    suite.addTest(TestScaledDotProductAttention("test_4d"))
    
    suite.addTest(TestDropHead("test_drop"))
    suite.addTest(TestDropHead("test_drop_all"))
    suite.addTest(TestDropHead("test_eval"))
    
    suite.addTest(TestMonoheadAttention("test_io"))
    
    suite.addTest(TestMultiheadAttention("test_io", same_qk=True))
    suite.addTest(TestMultiheadAttention("test_io", same_qk=False))
    
    return suite