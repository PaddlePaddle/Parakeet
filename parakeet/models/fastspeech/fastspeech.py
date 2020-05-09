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
import math
import numpy as np
import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
from parakeet.g2p.text.symbols import symbols
from parakeet.models.transformer_tts.utils import *
from parakeet.models.transformer_tts.post_convnet import PostConvNet
from parakeet.models.fastspeech.length_regulator import LengthRegulator
from parakeet.models.fastspeech.encoder import Encoder
from parakeet.models.fastspeech.decoder import Decoder


class FastSpeech(dg.Layer):
    def __init__(self, cfg, num_mels=80):
        """FastSpeech model.

        Args:
            cfg: the yaml configs used in FastSpeech model.
            num_mels (int, optional): the number of mel bands when calculating mel spectrograms. Defaults to 80.

        """
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(
            n_src_vocab=len(symbols) + 1,
            len_max_seq=cfg['max_seq_len'],
            n_layers=cfg['encoder_n_layer'],
            n_head=cfg['encoder_head'],
            d_k=cfg['hidden_size'] // cfg['encoder_head'],
            d_q=cfg['hidden_size'] // cfg['encoder_head'],
            d_model=cfg['hidden_size'],
            d_inner=cfg['encoder_conv1d_filter_size'],
            fft_conv1d_kernel=cfg['fft_conv1d_filter'],
            fft_conv1d_padding=cfg['fft_conv1d_padding'],
            dropout=0.1)
        self.length_regulator = LengthRegulator(
            input_size=cfg['hidden_size'],
            out_channels=cfg['duration_predictor_output_size'],
            filter_size=cfg['duration_predictor_filter_size'],
            dropout=cfg['dropout'])
        self.decoder = Decoder(
            len_max_seq=cfg['max_seq_len'],
            n_layers=cfg['decoder_n_layer'],
            n_head=cfg['decoder_head'],
            d_k=cfg['hidden_size'] // cfg['decoder_head'],
            d_q=cfg['hidden_size'] // cfg['decoder_head'],
            d_model=cfg['hidden_size'],
            d_inner=cfg['decoder_conv1d_filter_size'],
            fft_conv1d_kernel=cfg['fft_conv1d_filter'],
            fft_conv1d_padding=cfg['fft_conv1d_padding'],
            dropout=0.1)
        self.weight = fluid.ParamAttr(
            initializer=fluid.initializer.XavierInitializer())
        k = math.sqrt(1.0 / cfg['hidden_size'])
        self.bias = fluid.ParamAttr(initializer=fluid.initializer.Uniform(
            low=-k, high=k))
        self.mel_linear = dg.Linear(
            cfg['hidden_size'],
            num_mels * cfg['outputs_per_step'],
            param_attr=self.weight,
            bias_attr=self.bias, )
        self.postnet = PostConvNet(
            n_mels=num_mels,
            num_hidden=512,
            filter_size=5,
            padding=int(5 / 2),
            num_conv=5,
            outputs_per_step=cfg['outputs_per_step'],
            use_cudnn=True,
            dropout=0.1,
            batchnorm_last=True)

    def forward(self,
                character,
                text_pos,
                mel_pos=None,
                length_target=None,
                alpha=1.0):
        """
        Compute mel output from text character.
        
        Args:
            character (Variable): shape(B, T_text), dtype float32, the input text characters, 
                where T_text means the timesteps of input characters, 
            text_pos (Variable): shape(B, T_text), dtype int64, the input text position. 
            mel_pos (Variable, optional): shape(B, T_mel), dtype int64, the spectrum position, 
                where T_mel means the timesteps of input spectrum,  
            length_target (Variable, optional): shape(B, T_text), dtype int64, 
                the duration of phoneme compute from pretrained transformerTTS. Defaults to None. 
            alpha (float32, optional): The hyperparameter to determine the length of the expanded sequence 
                mel, thereby controlling the voice speed. Defaults to 1.0.

        Returns:
            mel_output (Variable): shape(B, T_mel, C), the mel output before postnet.
            mel_output_postnet (Variable): shape(B, T_mel, C), the mel output after postnet.
            duration_predictor_output (Variable): shape(B, T_text), the duration of phoneme compute with duration predictor. 
            enc_slf_attn_list (List[Variable]): len(enc_n_layers), the encoder self attention list. 
            dec_slf_attn_list (List[Variable]): len(dec_n_layers), the decoder self attention list.
        """

        encoder_output, enc_slf_attn_list = self.encoder(character, text_pos)
        if fluid.framework._dygraph_tracer()._train_mode:
            length_regulator_output, duration_predictor_output = self.length_regulator(
                encoder_output, target=length_target, alpha=alpha)
            decoder_output, dec_slf_attn_list = self.decoder(
                length_regulator_output, mel_pos)

            mel_output = self.mel_linear(decoder_output)
            mel_output_postnet = self.postnet(mel_output) + mel_output

            return mel_output, mel_output_postnet, duration_predictor_output, enc_slf_attn_list, dec_slf_attn_list
        else:
            length_regulator_output, decoder_pos = self.length_regulator(
                encoder_output, alpha=alpha)
            decoder_output, _ = self.decoder(length_regulator_output,
                                             decoder_pos)
            mel_output = self.mel_linear(decoder_output)
            mel_output_postnet = self.postnet(mel_output) + mel_output

            return mel_output, mel_output_postnet
