import numpy as np
import math
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers
import paddle.fluid as fluid
from parakeet.modules.multihead_attention import MultiheadAttention
from parakeet.modules.ffn import PositionwiseFeedForward

class FFTBlock(dg.Layer):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, filter_size, padding, dropout=0.2):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiheadAttention(d_model, d_k, d_v, num_head=n_head, is_bias=True, dropout=dropout, is_concat=False)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, filter_size =filter_size, padding =padding, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        """
        Feed Forward Transformer block in FastSpeech.
        
        Args:
            enc_input (Variable): Shape(B, T, C), dtype: float32. The embedding characters input. 
                T means the timesteps of input.
            non_pad_mask (Variable): Shape(B, T, 1), dtype: int64. The mask of sequence.
            slf_attn_mask (Variable): Shape(B, len_q, len_k), dtype: int64. The mask of self attention. 
                len_q means the sequence length of query, len_k means the sequence length of key.

        Returns:
            output (Variable), Shape(B, T, C), the output after self-attention & ffn.
            slf_attn (Variable), Shape(B * n_head, T, T), the self attention.
        """
        output, slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        output *= non_pad_mask

        output = self.pos_ffn(output)
        output *= non_pad_mask

        return output, slf_attn