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

        return mel_output, postnet_output, attn_probs, stop_preds, attns_enc, attns_dec
