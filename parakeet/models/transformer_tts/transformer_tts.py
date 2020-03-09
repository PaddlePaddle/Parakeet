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
        """TransformerTTS model.

        Args:
            config: the yaml configs used in TransformerTTS model.
        """
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
            characters (Variable): shape(B, T_text), dtype float32, the input character,
                where T_text means the timesteps of input text,
            mel_input (Variable): shape(B, T_mel, C), dtype float32, the input query of decoder,
                where T_mel means the timesteps of input spectrum,
            pos_text (Variable): shape(B, T_text), dtype int64, the characters position. 
            dec_slf_mask (Variable): shape(B, T_mel), dtype int64, the spectrum position. 
            mask (Variable): shape(B, T_mel, T_mel), dtype int64, the mask of decoder self attention.
            enc_slf_mask (Variable, optional): shape(B, T_text, T_text), dtype int64, the mask of encoder self attention. Defaults to None.
            enc_query_mask (Variable, optional): shape(B, T_text, 1), dtype int64, the query mask of encoder self attention. Defaults to None.
            dec_query_mask (Variable, optional): shape(B, T_mel, 1), dtype int64, the query mask of encoder-decoder attention. Defaults to None.
            dec_query_slf_mask (Variable, optional): shape(B, T_mel, 1), dtype int64, the query mask of decoder self attention. Defaults to None.
            enc_dec_mask (Variable, optional): shape(B, T_mel, T_text), dtype int64, query mask of encoder-decoder attention. Defaults to None.
                
        Returns:
            mel_output (Variable): shape(B, T_mel, C), the decoder output after mel linear projection.
            postnet_output (Variable): shape(B, T_mel, C), the decoder output after post mel network.
            stop_preds (Variable): shape(B, T_mel, 1), the stop tokens of output.
            attn_probs (list[Variable]): len(n_layers), the encoder-decoder attention list.
            attns_enc (list[Variable]): len(n_layers), the encoder self attention list.
            attns_dec (list[Variable]): len(n_layers), the decoder self attention list.
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
