import unittest
import numpy as np
import paddle
paddle.set_default_dtype("float64")
paddle.disable_static(paddle.CPUPlace())

from parakeet.models import transformer_tts as tts
from parakeet.modules import masking
from pprint import pprint

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
        enc_mask = masking.sequence_mask(enc_lengths, dtype=k.dtype)
        dec_padding_mask = masking.sequence_mask(dec_lengths, dtype=q.dtype)
        no_future_mask = masking.future_mask(32, dtype=q.dtype)
        dec_mask = masking.combine_mask(dec_padding_mask.unsqueeze(-1), no_future_mask)
        y, self_attn_weights, cross_attn_weights = net(q, k, v, enc_mask, dec_mask)
        
        self.assertTupleEqual(y.numpy().shape, (4, 32, 64))
        self.assertTupleEqual(self_attn_weights.numpy().shape, (4, 8, 32, 32))
        self.assertTupleEqual(cross_attn_weights.numpy().shape, (4, 8, 32, 24))
        
        
class TestTransformerTTS(unittest.TestCase):
    def setUp(self):
        net = tts.TransformerTTS(
            128, 0, 64, 80, 4, 128, 
            0.5,
            6, 6, 128, 128, 4, 
            3, 10, 0.5)
        self.net = net
        
    def test_encode_io(self):
        net = self.net
        
        text = paddle.randint(0, 128, [4, 176])
        lengths = paddle.to_tensor([176, 156, 174, 168])
        mask = masking.sequence_mask(lengths, dtype=text.dtype)
        text = text * mask
        
        encoded, attention_weights, encoder_mask = net.encode(text)
        print("output shapes:")
        print("encoded:", encoded.numpy().shape)
        print("encoder_attentions:", [item.shape for item in attention_weights])
        print("encoder_mask:", encoder_mask.numpy().shape)
        
    def test_all_io(self):
        net = self.net
        
        text = paddle.randint(0, 128, [4, 176])
        lengths = paddle.to_tensor([176, 156, 174, 168])
        mask = masking.sequence_mask(lengths, dtype=text.dtype)
        text = text * mask
        
        mel = paddle.randn([4, 189, 80])
        frames = paddle.to_tensor([189, 186, 179, 174])
        mask = masking.sequence_mask(frames, dtype=frames.dtype)
        mel = mel * mask.unsqueeze(-1)
        
        encoded, encoder_attention_weights, encoder_mask = net.encode(text)
        mel_output, mel_intermediate, cross_attention_weights, stop_logits = net.decode(encoded, mel, encoder_mask)
        
        print("output shapes:")
        print("encoder_output:", encoded.numpy().shape)
        print("encoder_attentions:", [item.shape for item in encoder_attention_weights])
        print("encoder_mask:", encoder_mask.numpy().shape)
        print("mel_output: ", mel_output.numpy().shape)
        print("mel_intermediate: ", mel_intermediate.numpy().shape)
        print("decoder_attentions:", [item.shape for item in cross_attention_weights])
        print("stop_logits:", stop_logits.numpy().shape)
        
    def test_predict_io(self):
        net = self.net
        net.eval()
        with paddle.no_grad():
            text = paddle.randint(0, 128, [176])
            decoder_output, encoder_attention_weights, cross_attention_weights = net.predict(text)
        
        print("output shapes:")
        print("mel_output: ", decoder_output.numpy().shape)
        print("encoder_attentions:", [item.shape for item in encoder_attention_weights])
        print("decoder_attentions:", [item.shape for item in cross_attention_weights])
        