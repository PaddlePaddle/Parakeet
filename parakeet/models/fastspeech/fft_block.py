# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import math
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers
import paddle.fluid as fluid
from parakeet.modules.multihead_attention import MultiheadAttention
from parakeet.modules.ffn import PositionwiseFeedForward


class FFTBlock(dg.Layer):
    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_q,
                 filter_size,
                 padding,
                 dropout=0.2):
        """Feed forward structure based on self-attention.

        Args:
            d_model (int): the dim of hidden layer in multihead attention.
            d_inner (int): the dim of hidden layer in ffn.
            n_head (int): the head number of multihead attention.
            d_k (int): the dim of key in multihead attention.
            d_q (int): the dim of query in multihead attention.
            filter_size (int): the conv kernel size.
            padding (int): the conv padding size.
            dropout (float, optional): dropout probability. Defaults to 0.2.
        """
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiheadAttention(
            d_model,
            d_k,
            d_q,
            num_head=n_head,
            is_bias=True,
            dropout=dropout,
            is_concat=False)
        self.pos_ffn = PositionwiseFeedForward(
            d_model,
            d_inner,
            filter_size=filter_size,
            padding=padding,
            dropout=dropout)

    def forward(self, enc_input, non_pad_mask, slf_attn_mask=None):
        """
        Feed forward block of FastSpeech
        
        Args:
            enc_input (Variable): shape(B, T, C), dtype float32, the embedding characters input, 
                where T means the timesteps of input.   
            non_pad_mask (Variable): shape(B, T, 1), dtype int64, the mask of sequence.
            slf_attn_mask (Variable, optional): shape(B, len_q, len_k), dtype int64, the mask of self attention,
                where len_q means the sequence length of query and len_k means the sequence length of key. Defaults to None. 
                     
        Returns:
            output (Variable): shape(B, T, C), the output after self-attention & ffn. 
            slf_attn (Variable): shape(B * n_head, T, T), the self attention.
        """
        output, slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)

        output *= non_pad_mask

        output = self.pos_ffn(output)
        output *= non_pad_mask

        return output, slf_attn
