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
    def __init__(self,
                 embedding_size,
                 num_hidden,
                 encoder_num_head=4,
                 encoder_n_layers=3,
                 n_mels=80,
                 outputs_per_step=1,
                 decoder_num_head=4,
                 decoder_n_layers=3):
        """TransformerTTS model.

        Args:
            embedding_size (int): the size of position embedding.
            num_hidden (int): the size of hidden layer in network.
            encoder_num_head (int, optional): the head number of multihead attention in encoder. Defaults to 4.
            encoder_n_layers (int, optional): the layers number of multihead attention in encoder. Defaults to 3.
            n_mels (int, optional): the number of mel bands when calculating mel spectrograms. Defaults to 80.
            outputs_per_step (int, optional): the num of output frames per step . Defaults to 1.
            decoder_num_head (int, optional): the head number of multihead attention in decoder. Defaults to 4.
            decoder_n_layers (int, optional): the layers number of multihead attention in decoder. Defaults to 3.
        """
        super(TransformerTTS, self).__init__()
        self.encoder = Encoder(embedding_size, num_hidden, encoder_num_head,
                               encoder_n_layers)
        self.decoder = Decoder(num_hidden, n_mels, outputs_per_step,
                               decoder_num_head, decoder_n_layers)

    def forward(self, characters, mel_input, pos_text, pos_mel):
        """
        TransformerTTS network.
        
        Args:
            characters (Variable): shape(B, T_text), dtype float32, the input character,
                where T_text means the timesteps of input text,
            mel_input (Variable): shape(B, T_mel, C), dtype float32, the input query of decoder,
                where T_mel means the timesteps of input spectrum,
            pos_text (Variable): shape(B, T_text), dtype int64, the characters position. 
                
        Returns:
            mel_output (Variable): shape(B, T_mel, C), the decoder output after mel linear projection.
            postnet_output (Variable): shape(B, T_mel, C), the decoder output after post mel network.
            stop_preds (Variable): shape(B, T_mel, 1), the stop tokens of output.
            attn_probs (list[Variable]): len(n_layers), the encoder-decoder attention list.
            attns_enc (list[Variable]): len(n_layers), the encoder self attention list.
            attns_dec (list[Variable]): len(n_layers), the decoder self attention list.
        """
        key, attns_enc, query_mask = self.encoder(characters, pos_text)

        mel_output, postnet_output, attn_probs, stop_preds, attns_dec = self.decoder(
            key, key, mel_input, pos_mel, query_mask)
        return mel_output, postnet_output, attn_probs, stop_preds, attns_enc, attns_dec
