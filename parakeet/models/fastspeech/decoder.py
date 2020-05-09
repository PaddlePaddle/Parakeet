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


class Decoder(dg.Layer):
    def __init__(self,
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
        """Decoder layer of FastSpeech.

        Args:
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
        super(Decoder, self).__init__()

        n_position = len_max_seq + 1
        self.n_head = n_head
        self.pos_inp = get_sinusoid_encoding_table(
            n_position, d_model, padding_idx=0)
        self.position_enc = dg.Embedding(
            size=[n_position, d_model],
            padding_idx=0,
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

    def forward(self, enc_seq, enc_pos):
        """
        Compute decoder outputs.
        
        Args:
            enc_seq (Variable): shape(B, T_mel, C), dtype float32,
                the output of length regulator, where T_mel means the timesteps of input spectrum.
            enc_pos (Variable): shape(B, T_mel), dtype int64, 
                the spectrum position.

        Returns:
            dec_output (Variable): shape(B, T_mel, C), the decoder output.
            dec_slf_attn_list (list[Variable]): len(n_layers), the decoder self attention list.
        """
        dec_slf_attn_list = []
        if fluid.framework._dygraph_tracer()._train_mode:
            slf_attn_mask = get_dec_attn_key_pad_mask(enc_pos, self.n_head,
                                                      enc_seq.dtype)

        else:
            len_q = enc_seq.shape[1]
            slf_attn_mask = layers.triu(
                layers.ones(
                    shape=[len_q, len_q], dtype=enc_seq.dtype),
                diagonal=1)
            slf_attn_mask = layers.cast(
                slf_attn_mask != 0, dtype=enc_seq.dtype) * -1e30

        non_pad_mask = get_non_pad_mask(enc_pos, 1, enc_seq.dtype)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            dec_slf_attn_list += [dec_slf_attn]

        return dec_output, dec_slf_attn_list
