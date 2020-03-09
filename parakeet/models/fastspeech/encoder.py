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
import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
from parakeet.models.transformer_tts.utils import *
from parakeet.models.fastspeech.fft_block import FFTBlock


class Encoder(dg.Layer):
    def __init__(self,
                 n_src_vocab,
                 len_max_seq,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 d_model,
                 d_inner,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout=0.1):
        super(Encoder, self).__init__()
        n_position = len_max_seq + 1
        self.n_head = n_head

        self.src_word_emb = dg.Embedding(
            size=[n_src_vocab, d_model],
            padding_idx=0,
            param_attr=fluid.initializer.Normal(
                loc=0.0, scale=1.0))
        self.pos_inp = get_sinusoid_encoding_table(
            n_position, d_model, padding_idx=0)
        self.position_enc = dg.Embedding(
            size=[n_position, d_model],
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    self.pos_inp),
                trainable=False))
        self.layer_stack = [
            FFTBlock(
                d_model,
                d_inner,
                n_head,
                d_k,
                d_v,
                fft_conv1d_kernel,
                fft_conv1d_padding,
                dropout=dropout) for _ in range(n_layers)
        ]
        for i, layer in enumerate(self.layer_stack):
            self.add_sublayer('fft_{}'.format(i), layer)

    def forward(self, character, text_pos, non_pad_mask, slf_attn_mask=None):
        """
        Encoder layer of FastSpeech.
        Args:
            character (Variable): The input text characters. 
                Shape: (B, T_text), T_text means the timesteps of input characters,
                dtype: float32.
            text_pos (Variable): The input text position. 
                Shape: (B, T_text), dtype: int64.
            non_pad_mask (Variable): the mask with non pad.
                Shape: (B, T_text, 1),
                dtype: int64.
            slf_attn_mask (Variable, optional): the mask of input characters. Defaults to None.
                Shape: (B, T_text, T_text),
                dtype: int64.
                
        Returns:
            enc_output (Variable), the encoder output. Shape(B, T_text, C)
            non_pad_mask (Variable), the mask with non pad. Shape(B, T_text, 1)
            enc_slf_attn_list (list[Variable]), the encoder self attention list.
                Len: n_layers.
        """
        enc_slf_attn_list = []
        slf_attn_mask = layers.expand(slf_attn_mask, [self.n_head, 1, 1])

        # -- Forward
        enc_output = self.src_word_emb(character) + self.position_enc(
            text_pos)  #(N, T, C)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            enc_slf_attn_list += [enc_slf_attn]

        return enc_output, enc_slf_attn_list
