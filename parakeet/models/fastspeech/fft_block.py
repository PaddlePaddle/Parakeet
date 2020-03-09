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
                 d_v,
                 filter_size,
                 padding,
                 dropout=0.2):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiheadAttention(
            d_model,
            d_k,
            d_v,
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
        Feed Forward Transformer block in FastSpeech.
        
        Args:
            enc_input (Variable): The embedding characters input. 
                Shape: (B, T, C), T means the timesteps of input, dtype: float32.   
            non_pad_mask (Variable): The mask of sequence.
                Shape: (B, T, 1), dtype: int64.
            slf_attn_mask (Variable, optional): The mask of self attention. Defaults to None.
                Shape(B, len_q, len_k), len_q means the sequence length of query, 
                len_k means the sequence length of key, dtype: int64.   
                     
        Returns:
            output (Variable), the output after self-attention & ffn. Shape: (B, T, C).
            slf_attn (Variable), the self attention. Shape: (B * n_head, T, T),
        """
        output, slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)

        output *= non_pad_mask

        output = self.pos_ffn(output)
        output *= non_pad_mask

        return output, slf_attn
