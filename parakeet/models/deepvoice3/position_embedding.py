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

from __future__ import division
import numpy as np
from paddle import fluid
import paddle.fluid.layers as F
import paddle.fluid.dygraph as dg


def lookup(weight, indices, padding_idx):
    out = fluid.core.ops.lookup_table_v2(
        weight, indices, 'is_sparse', False, 'is_distributed', False,
        'remote_prefetch', False, 'padding_idx', padding_idx)
    return out


def compute_position_embedding_single_speaker(radians, speaker_position_rate):
    """Compute sin/cos interleaved matrix from the radians.
    
    Arg:
        radians (Variable): shape(n_vocab, embed_dim), dtype float32, the radians matrix.
        speaker_position_rate (float or Variable): float or Variable of shape(1, ), speaker positioning rate.
    
    Returns:
        Variable: shape(n_vocab, embed_dim), the sin, cos interleaved matrix.
    """
    _, embed_dim = radians.shape
    scaled_radians = radians * speaker_position_rate

    odd_mask = (np.arange(embed_dim) % 2).astype(np.float32)
    odd_mask = dg.to_variable(odd_mask)

    out = odd_mask * F.cos(scaled_radians) \
        + (1 - odd_mask) * F.sin(scaled_radians)
    return out


def compute_position_embedding(radians, speaker_position_rate):
    """Compute sin/cos interleaved matrix from the radians.
    
    Arg:
        radians (Variable): shape(n_vocab, embed_dim), dtype float32, the radians matrix.
        speaker_position_rate (Variable): shape(B, ), speaker positioning rate.
    
    Returns:
        Variable: shape(B, n_vocab, embed_dim), the sin, cos interleaved matrix.
    """
    _, embed_dim = radians.shape
    batch_size = speaker_position_rate.shape[0]
    scaled_radians = F.elementwise_mul(
        F.expand(F.unsqueeze(radians, [0]), [batch_size, 1, 1]),
        speaker_position_rate,
        axis=0)

    odd_mask = (np.arange(embed_dim) % 2).astype(np.float32)
    odd_mask = dg.to_variable(odd_mask)

    out = odd_mask * F.cos(scaled_radians) \
        + (1 - odd_mask) * F.sin(scaled_radians)
    out = F.concat(
        [F.zeros((batch_size, 1, embed_dim), radians.dtype), out[:, 1:, :]],
        axis=1)
    return out


def position_encoding_init(n_position,
                           d_pos_vec,
                           position_rate=1.0,
                           padding_idx=None):
    """Init the position encoding.

    Args:
        n_position (int): max position, vocab size for position embedding.
        d_pos_vec (int): position embedding size.
        position_rate (float, optional): position rate (this should only be used when all the utterances are from one speaker.). Defaults to 1.0.
        padding_idx (int, optional): padding index for the position embedding(it is set as 0 internally if not provided.). Defaults to None.

    Returns:
        [type]: [description]
    """
    # init the position encoding table
    # keep idx 0 for padding token position encoding zero vector
    # CAUTION: it is radians here, sin and cos are not applied
    indices_range = np.expand_dims(np.arange(n_position), -1)
    embed_range = 2 * (np.arange(d_pos_vec) // 2)
    radians = position_rate \
            * indices_range \
            / np.power(1.e4, embed_range / d_pos_vec)
    if padding_idx is not None:
        radians[padding_idx] = 0.
    return radians


class PositionEmbedding(dg.Layer):
    def __init__(self, n_position, d_pos_vec, position_rate=1.0):
        """Position Embedding for Deep Voice 3.

        Args:
            n_position (int): max position, vocab size for position embedding.
            d_pos_vec (int): position embedding size.
            position_rate (float, optional): position rate (this should only be used when all the utterances are from one speaker.). Defaults to 1.0.
        """
        super(PositionEmbedding, self).__init__()
        self.weight = self.create_parameter((n_position, d_pos_vec))
        self.weight.set_value(
            position_encoding_init(n_position, d_pos_vec, position_rate)
            .astype("float32"))

    def forward(self, indices, speaker_position_rate=None):
        """
        Args:
            indices (Variable): shape (B, T), dtype: int64, position
                indices, where B means the batch size, T means the time steps.
            speaker_position_rate (Variable | float, optional), position
                rate. It can be a float point number or a Variable with 
                shape (1,), then this speaker_position_rate is used for every 
                example. It can also be a Variable with shape (B, ), which 
                contains a speaker position rate for each utterance.
        Returns:
            out (Variable): shape(B, T, C_pos), dtype float32, position embedding, where C_pos 
                means position embedding size.
        """
        batch_size, time_steps = indices.shape

        if isinstance(speaker_position_rate, float) or \
            (isinstance(speaker_position_rate, fluid.framework.Variable)
            and list(speaker_position_rate.shape) == [1]):
            temp_weight = compute_position_embedding_single_speaker(
                self.weight, speaker_position_rate)
            out = lookup(temp_weight, indices, 0)
            return out

        assert len(speaker_position_rate.shape) == 1 and \
            list(speaker_position_rate.shape) == [batch_size]

        weight = compute_position_embedding(self.weight,
                                            speaker_position_rate)  # (B, V, C)
        # make indices for gather_nd
        batch_id = F.expand(
            F.unsqueeze(
                F.range(
                    0, batch_size, 1, dtype="int64"), [1]), [1, time_steps])
        # (B, T, 2)
        gather_nd_id = F.stack([batch_id, indices], -1)
        out = F.gather_nd(weight, gather_nd_id)
        return out
