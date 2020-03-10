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
from collections import namedtuple

import paddle.fluid.layers as F
import paddle.fluid.initializer as I
import paddle.fluid.dygraph as dg

from parakeet.modules.weight_norm import Conv1D, Linear
from parakeet.models.deepvoice3.conv1dglu import Conv1DGLU

ConvSpec = namedtuple("ConvSpec", ["out_channels", "filter_size", "dilation"])


class Encoder(dg.Layer):
    def __init__(self,
                 n_vocab,
                 embed_dim,
                 n_speakers,
                 speaker_dim,
                 padding_idx=None,
                 embedding_weight_std=0.1,
                 convolutions=(ConvSpec(64, 5, 1), ) * 7,
                 dropout=0.):
        """Encoder of Deep Voice 3.

        Args:
            n_vocab (int): vocabulary size of the text embedding.
            embed_dim (int): embedding size of the text embedding.
            n_speakers (int): number of speakers.
            speaker_dim (int): speaker embedding size.
            padding_idx (int, optional): padding index of text embedding. Defaults to None.
            embedding_weight_std (float, optional): standard deviation of the embedding weights when intialized. Defaults to 0.1.
            convolutions (Iterable[ConvSpec], optional): specifications of the convolutional layers. ConvSpec is a namedtuple of output channels, filter_size and dilation. Defaults to (ConvSpec(64, 5, 1), )*7.
            dropout (float, optional): dropout probability. Defaults to 0..
        """
        super(Encoder, self).__init__()
        self.embedding_weight_std = embedding_weight_std
        self.embed = dg.Embedding(
            (n_vocab, embed_dim),
            padding_idx=padding_idx,
            param_attr=I.Normal(scale=embedding_weight_std))

        self.dropout = dropout
        if n_speakers > 1:
            std = np.sqrt((1 - dropout) / speaker_dim)
            self.sp_proj1 = Linear(
                speaker_dim,
                embed_dim,
                act="softsign",
                param_attr=I.Normal(scale=std))
            self.sp_proj2 = Linear(
                speaker_dim,
                embed_dim,
                act="softsign",
                param_attr=I.Normal(scale=std))
        self.n_speakers = n_speakers

        self.convolutions = dg.LayerList()
        in_channels = embed_dim
        std_mul = 1.0
        for (out_channels, filter_size, dilation) in convolutions:
            # 1 * 1 convolution & relu
            if in_channels != out_channels:
                std = np.sqrt(std_mul / in_channels)
                self.convolutions.append(
                    Conv1D(
                        in_channels,
                        out_channels,
                        1,
                        act="relu",
                        param_attr=I.Normal(scale=std)))
                in_channels = out_channels
                std_mul = 2.0

            self.convolutions.append(
                Conv1DGLU(
                    n_speakers,
                    speaker_dim,
                    in_channels,
                    out_channels,
                    filter_size,
                    dilation,
                    std_mul,
                    dropout,
                    causal=False,
                    residual=True))
            in_channels = out_channels
            std_mul = 4.0

        std = np.sqrt(std_mul * (1 - dropout) / in_channels)
        self.convolutions.append(
            Conv1D(
                in_channels, embed_dim, 1, param_attr=I.Normal(scale=std)))

    def forward(self, x, speaker_embed=None):
        """
        Encode text sequence.
        
        Args:
            x (Variable): shape(B, T_enc), dtype: int64. Ihe input text indices. T_enc means the timesteps of decoder input x.
            speaker_embed (Variable, optional): shape(B, C_sp), dtype float32, speaker embeddings. This arg is not None only when the model is a multispeaker model.

        Returns:
            keys (Variable), Shape(B, T_enc, C_emb), dtype float32, the encoded epresentation for keys, where C_emb menas the text embedding size.
            values (Variable), Shape(B, T_enc, C_emb), dtype float32, the encoded representation for values.
        """
        x = self.embed(x)
        x = F.dropout(
            x, self.dropout, dropout_implementation="upscale_in_train")
        x = F.transpose(x, [0, 2, 1])

        if self.n_speakers > 1 and speaker_embed is not None:
            speaker_embed = F.dropout(
                speaker_embed,
                self.dropout,
                dropout_implementation="upscale_in_train")
            x = F.elementwise_add(x, self.sp_proj1(speaker_embed), axis=0)

        input_embed = x
        for layer in self.convolutions:
            if isinstance(layer, Conv1DGLU):
                x = layer(x, speaker_embed)
            else:
                # layer is a Conv1D with (1,) filter wrapped by WeightNormWrapper
                x = layer(x)

        if self.n_speakers > 1 and speaker_embed is not None:
            x = F.elementwise_add(x, self.sp_proj2(speaker_embed), axis=0)

        keys = x  # (B, C, T)
        values = F.scale(input_embed + x, scale=np.sqrt(0.5))
        keys = F.transpose(keys, [0, 2, 1])
        values = F.transpose(values, [0, 2, 1])
        return keys, values
