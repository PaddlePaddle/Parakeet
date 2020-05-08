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
                 d_q,
                 d_model,
                 d_inner,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout=0.1):
        """Encoder layer of FastSpeech.

        Args:
            n_src_vocab (int): the number of source vocabulary.
            len_max_seq (int): the max mel len of sequence.
            n_layers (int): the layers number of FFTBlock.
            n_head (int): the head number of multihead attention.
            d_k (int): the dim of key in multihead attention.
            d_q (int): the dim of query in multihead attention.
            d_model (int): the dim of hidden layer in multihead attention.
            d_inner (int): the dim of hidden layer in ffn.
            fft_conv1d_kernel (int): the conv kernel size in FFTBlock.
            fft_conv1d_padding (int): the conv padding size in FFTBlock.
            dropout (float, optional): dropout probability of FFTBlock. Defaults to 0.1.
        """
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
                d_q,
                fft_conv1d_kernel,
                fft_conv1d_padding,
                dropout=dropout) for _ in range(n_layers)
        ]
        for i, layer in enumerate(self.layer_stack):
            self.add_sublayer('fft_{}'.format(i), layer)

    def forward(self, character, text_pos):
        """
        Encode text sequence.

        Args:
            character (Variable): shape(B, T_text), dtype float32, the input text characters, 
                where T_text means the timesteps of input characters,
            text_pos (Variable): shape(B, T_text), dtype int64, the input text position. 
        
        Returns:
            enc_output (Variable): shape(B, T_text, C), the encoder output. 
            enc_slf_attn_list (list[Variable]): len(n_layers), the encoder self attention list.
        """
        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(character) + self.position_enc(
            text_pos)  #(N, T, C)

        slf_attn_mask = get_attn_key_pad_mask(text_pos, self.n_head,
                                              enc_output.dtype)
        non_pad_mask = get_non_pad_mask(text_pos, 1, enc_output.dtype)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            enc_slf_attn_list += [enc_slf_attn]

        return enc_output, enc_slf_attn_list
