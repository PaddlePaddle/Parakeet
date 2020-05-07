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
import librosa
import os, copy
from scipy import signal
import paddle.fluid.layers as layers


def get_positional_table(d_pos_vec, n_position=1024):
    position_enc = np.array(
        [[pos / np.power(10000, 2 * i / d_pos_vec) for i in range(d_pos_vec)]
         if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return position_enc


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return sinusoid_table


def get_non_pad_mask(seq, num_head, dtype):
    mask = layers.cast(seq != 0, dtype=dtype)
    mask = layers.unsqueeze(mask, axes=[-1])
    mask = layers.expand(mask, [num_head, 1, 1])
    return mask


def get_attn_key_pad_mask(seq_k, num_head, dtype):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    padding_mask = layers.cast(seq_k == 0, dtype=dtype) * -1e30
    padding_mask = layers.unsqueeze(padding_mask, axes=[1])
    padding_mask = layers.expand(padding_mask, [num_head, 1, 1])
    return padding_mask


def get_dec_attn_key_pad_mask(seq_k, num_head, dtype):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    padding_mask = layers.cast(seq_k == 0, dtype=dtype)
    padding_mask = layers.unsqueeze(padding_mask, axes=[1])
    len_k = seq_k.shape[1]
    triu = layers.triu(
        layers.ones(
            shape=[len_k, len_k], dtype=dtype), diagonal=1)
    padding_mask = padding_mask + triu
    padding_mask = layers.cast(
        padding_mask != 0, dtype=dtype) * -1e30  #* (-2**32 + 1)
    padding_mask = layers.expand(padding_mask, [num_head, 1, 1])
    return padding_mask


def guided_attention(N, T, g=0.2):
    '''Guided attention. Refer to page 3 on the paper.'''
    W = np.zeros((N, T), dtype=np.float32)
    for n_pos in range(W.shape[0]):
        for t_pos in range(W.shape[1]):
            W[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(T) - n_pos / float(N))
                                         **2 / (2 * g * g))
    return W


def cross_entropy(input, label, position_weight=1.0, epsilon=1e-30):
    output = -1 * label * layers.log(input + epsilon) - (
        1 - label) * layers.log(1 - input + epsilon)
    output = output * (label * (position_weight - 1) + 1)

    return layers.reduce_sum(output, dim=[0, 1])
