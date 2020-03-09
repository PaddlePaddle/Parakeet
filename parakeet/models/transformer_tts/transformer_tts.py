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
from parakeet.models.transformer_tts.encoder import Encoder
from parakeet.models.transformer_tts.decoder import Decoder


class TransformerTTS(dg.Layer):
    def __init__(self, config):
        super(TransformerTTS, self).__init__()
        self.encoder = Encoder(config['embedding_size'], config['hidden_size'])
        self.decoder = Decoder(config['hidden_size'], config)
        self.config = config

    def forward(self,
                characters,
                mel_input,
                pos_text,
                pos_mel,
                dec_slf_mask,
                enc_slf_mask=None,
                enc_query_mask=None,
                enc_dec_mask=None,
                dec_query_slf_mask=None,
                dec_query_mask=None):
        """
        TransformerTTS network.
        Args:
            characters (Variable): The input character.
                Shape: (B, T_text), T_text means the timesteps of input text,
                dtype: float32. 
            mel_input (Variable): The input query of decoder.
                Shape: (B, T_mel, C), T_mel means the timesteps of input spectrum,
                dtype: float32.
            pos_text (Variable): The characters position. 
                Shape: (B, T_text), dtype: int64.
            dec_slf_mask (Variable): The spectrum position. 
                Shape: (B, T_mel), dtype: int64.
            mask (Variable): the mask of decoder self attention.
                Shape: (B, T_mel, T_mel), dtype: int64.
            enc_slf_mask (Variable, optional): the mask of encoder self attention. Defaults to None.
                Shape: (B, T_text, T_text), dtype: int64.
            enc_query_mask (Variable, optional): the query mask of encoder self attention. Defaults to None.
                Shape: (B, T_text, 1), dtype: int64.
            dec_query_mask (Variable, optional): the query mask of encoder-decoder attention. Defaults to None.
                Shape: (B, T_mel, 1), dtype: int64.
            dec_query_slf_mask (Variable, optional): the query mask of decoder self attention. Defaults to None.
                Shape: (B, T_mel, 1), dtype: int64.
            enc_dec_mask (Variable, optional): query mask of encoder-decoder attention. Defaults to None.
                Shape: (B, T_mel, T_text), dtype: int64.
                
        Returns:
            mel_output (Variable): the decoder output after mel linear projection.
                Shape: (B, T_mel, C).
            postnet_output (Variable): the decoder output after post mel network.
                Shape: (B, T_mel, C).
            stop_preds (Variable): the stop tokens of output.
                Shape: (B, T_mel, 1)
            attn_probs (list[Variable]): the encoder-decoder attention list.
                Len: n_layers.
            attns_enc (list[Variable]): the encoder self attention list.
                Len: n_layers.
            attns_dec (list[Variable]): the decoder self attention list.
                Len: n_layers.
        """
        key, attns_enc = self.encoder(
            characters, pos_text, mask=enc_slf_mask, query_mask=enc_query_mask)

        mel_output, postnet_output, attn_probs, stop_preds, attns_dec = self.decoder(
            key,
            key,
            mel_input,
            pos_mel,
            mask=dec_slf_mask,
            zero_mask=enc_dec_mask,
            m_self_mask=dec_query_slf_mask,
            m_mask=dec_query_mask)
        return mel_output, postnet_output, attn_probs, stop_preds, attns_enc, attns_dec
