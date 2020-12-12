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
import paddle
from paddle import nn
from paddle.nn import functional as F
import parakeet
from parakeet.modules.conv import Conv1dBatchNorm
from parakeet.modules.attention import LocationSensitiveAttention
from parakeet.modules import masking
from parakeet.utils import checkpoint

__all__ = ["Tacotron2", "Tacotron2Loss"]


class DecoderPreNet(nn.Layer):
    def __init__(self,
                 d_input: int,
                 d_hidden: int,
                 d_output: int,
                 dropout_rate: float=0.2):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(d_input, d_hidden, bias_attr=False)
        self.linear2 = nn.Linear(d_hidden, d_output, bias_attr=False)

    def forward(self, x):
        x = F.dropout(F.relu(self.linear1(x)), self.dropout_rate)
        output = F.dropout(F.relu(self.linear2(x)), self.dropout_rate)
        return output


class DecoderPostNet(nn.Layer):
    def __init__(self,
                 d_mels: int=80,
                 d_hidden: int=512,
                 kernel_size: int=5,
                 padding: int=0,
                 num_layers: int=5,
                 dropout: float=0.1):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        self.conv_batchnorms = nn.LayerList()
        k = math.sqrt(1.0 / (d_mels * kernel_size))
        self.conv_batchnorms.append(
            Conv1dBatchNorm(
                d_mels,
                d_hidden,
                kernel_size=kernel_size,
                padding=padding,
                bias_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(
                    low=-k, high=k)),
                data_format='NLC'))

        k = math.sqrt(1.0 / (d_hidden * kernel_size))
        self.conv_batchnorms.extend([
            Conv1dBatchNorm(
                d_hidden,
                d_hidden,
                kernel_size=kernel_size,
                padding=padding,
                bias_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(
                    low=-k, high=k)),
                data_format='NLC') for i in range(1, num_layers - 1)
        ])

        self.conv_batchnorms.append(
            Conv1dBatchNorm(
                d_hidden,
                d_mels,
                kernel_size=kernel_size,
                padding=padding,
                bias_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(
                    low=-k, high=k)),
                data_format='NLC'))

    def forward(self, input):
        for i in range(len(self.conv_batchnorms) - 1):
            input = F.dropout(
                F.tanh(self.conv_batchnorms[i](input), self.dropout))
        input = F.dropout(self.conv_batchnorms[self.num_layers - 1](input),
                          self.dropout)
        return input


class Tacotron2Encoder(nn.Layer):
    def __init__(self,
                 d_hidden: int,
                 conv_layers: int,
                 kernel_size: int,
                 p_dropout: float):
        super().__init__()

        k = math.sqrt(1.0 / (d_hidden * kernel_size))
        self.conv_batchnorms = paddle.nn.LayerList([
            Conv1dBatchNorm(
                d_hidden,
                d_hidden,
                kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                bias_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(
                    low=-k, high=k)),
                data_format='NLC') for i in range(conv_layers)
        ])
        self.p_dropout = p_dropout

        self.hidden_size = int(d_hidden / 2)
        self.lstm = nn.LSTM(
            d_hidden, self.hidden_size, direction="bidirectional")

    def forward(self, x, input_lens=None):
        for conv_batchnorm in self.conv_batchnorms:
            x = F.dropout(F.relu(conv_batchnorm(x)),
                          self.p_dropout)  #(B, T, C)

        output, _ = self.lstm(inputs=x, sequence_length=input_lens)
        return output


class Tacotron2Decoder(nn.Layer):
    def __init__(self,
                 d_mels: int,
                 reduction_factor: int,
                 d_encoder: int,
                 d_prenet: int,
                 d_attention_rnn: int,
                 d_decoder_rnn: int,
                 d_attention: int,
                 attention_filters: int,
                 attention_kernel_size: int,
                 p_prenet_dropout: float,
                 p_attention_dropout: float,
                 p_decoder_dropout: float):
        super().__init__()
        self.d_mels = d_mels
        self.reduction_factor = reduction_factor
        self.d_encoder = d_encoder
        self.d_attention_rnn = d_attention_rnn
        self.d_decoder_rnn = d_decoder_rnn
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout

        self.prenet = DecoderPreNet(
            d_mels * reduction_factor,
            d_prenet,
            d_prenet,
            dropout_rate=p_prenet_dropout)

        self.attention_rnn = nn.LSTMCell(d_prenet + d_encoder, d_attention_rnn)

        self.attention_layer = LocationSensitiveAttention(
            d_attention_rnn, d_encoder, d_attention, attention_filters,
            attention_kernel_size)
        self.decoder_rnn = nn.LSTMCell(d_attention_rnn + d_encoder,
                                       d_decoder_rnn)
        self.linear_projection = nn.Linear(d_decoder_rnn + d_encoder,
                                           d_mels * reduction_factor)
        self.stop_layer = nn.Linear(d_decoder_rnn + d_encoder, 1)

    def _initialize_decoder_states(self, key):
        batch_size = key.shape[0]
        MAX_TIME = key.shape[1]

        self.attention_hidden = paddle.zeros(
            shape=[batch_size, self.d_attention_rnn], dtype=key.dtype)
        self.attention_cell = paddle.zeros(
            shape=[batch_size, self.d_attention_rnn], dtype=key.dtype)

        self.decoder_hidden = paddle.zeros(
            shape=[batch_size, self.d_decoder_rnn], dtype=key.dtype)
        self.decoder_cell = paddle.zeros(
            shape=[batch_size, self.d_decoder_rnn], dtype=key.dtype)

        self.attention_weights = paddle.zeros(
            shape=[batch_size, MAX_TIME], dtype=key.dtype)
        self.attention_weights_cum = paddle.zeros(
            shape=[batch_size, MAX_TIME], dtype=key.dtype)
        self.attention_context = paddle.zeros(
            shape=[batch_size, self.d_encoder], dtype=key.dtype)

        self.key = key  #[B, T, C]
        self.processed_key = self.attention_layer.key_layer(key)  #[B, T, C]

    def _decode(self, query):
        cell_input = paddle.concat([query, self.attention_context], axis=-1)

        # The first lstm layer
        _, (self.attention_hidden, self.attention_cell) = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(self.attention_hidden,
                                          self.p_attention_dropout)

        # Loaction sensitive attention
        attention_weights_cat = paddle.stack(
            [self.attention_weights, self.attention_weights_cum], axis=-1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.processed_key, self.key,
            attention_weights_cat, self.mask)
        self.attention_weights_cum += self.attention_weights

        # The second lstm layer
        decoder_input = paddle.concat(
            [self.attention_hidden, self.attention_context], axis=-1)
        _, (self.decoder_hidden, self.decoder_cell) = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, p=self.p_decoder_dropout)

        # decode output one step
        decoder_hidden_attention_context = paddle.concat(
            [self.decoder_hidden, self.attention_context], axis=-1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)
        stop_logit = self.stop_layer(decoder_hidden_attention_context)
        return decoder_output, stop_logit, self.attention_weights

    def forward(self, keys, querys, mask):
        querys = paddle.reshape(
            querys,
            [querys.shape[0], querys.shape[1] // self.reduction_factor, -1])
        querys = paddle.concat(
            [
                paddle.zeros(
                    shape=[
                        querys.shape[0], 1,
                        querys.shape[-1] * self.reduction_factor
                    ],
                    dtype=querys.dtype), querys
            ],
            axis=1)
        querys = self.prenet(querys)

        self._initialize_decoder_states(keys)
        self.mask = mask

        mel_outputs, stop_logits, alignments = [], [], []
        while len(mel_outputs) < querys.shape[
                1] - 1:  # Ignore the last time step
            query = querys[:, len(mel_outputs), :]
            mel_output, stop_logit, attention_weights = self._decode(query)
            mel_outputs += [mel_output]
            stop_logits += [stop_logit]
            alignments += [attention_weights]

        alignments = paddle.stack(alignments, axis=1)
        stop_logits = paddle.concat(stop_logits, axis=1)
        mel_outputs = paddle.stack(mel_outputs, axis=1)

        return mel_outputs, stop_logits, alignments

    def infer(self, key, stop_threshold=0.5, max_decoder_steps=1000):
        decoder_input = paddle.zeros(
            shape=[key.shape[0], self.d_mels * self.reduction_factor],
            dtype=key.dtype)  #[B, C]

        self._initialize_decoder_states(key)
        self.mask = None

        mel_outputs, stop_logits, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, stop_logit, alignment = self._decode(decoder_input)

            mel_outputs += [mel_output]
            stop_logits += [stop_logit]
            alignments += [alignment]

            if F.sigmoid(stop_logit) > stop_threshold:
                break
            elif len(mel_outputs) == max_decoder_steps:
                print("Warning! Reached max decoder steps!!!")
                break

            decoder_input = mel_output

        alignments = paddle.stack(alignments, axis=1)
        stop_logits = paddle.concat(stop_logits, axis=1)
        mel_outputs = paddle.stack(mel_outputs, axis=1)

        return mel_outputs, stop_logits, alignments


class Tacotron2(nn.Layer):
    """
    Tacotron2 module for end-to-end text-to-speech (E2E-TTS).

    This is a module of Spectrogram prediction network in Tacotron2 described
    in `Natural TTS Synthesis
    by Conditioning WaveNet on Mel Spectrogram Predictions`_,
    which converts the sequence of characters
    into the sequence of mel spectrogram.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884
    """

    def __init__(self,
                 frontend: parakeet.frontend.Phonetics,
                 d_mels: int=80,
                 d_encoder: int=512,
                 encoder_conv_layers: int=3,
                 encoder_kernel_size: int=5,
                 d_prenet: int=256,
                 d_attention_rnn: int=1024,
                 d_decoder_rnn: int=1024,
                 attention_filters: int=32,
                 attention_kernel_size: int=31,
                 d_attention: int=128,
                 d_postnet: int=512,
                 postnet_kernel_size: int=5,
                 postnet_conv_layers: int=5,
                 reduction_factor: int=1,
                 p_encoder_dropout: float=0.5,
                 p_prenet_dropout: float=0.5,
                 p_attention_dropout: float=0.1,
                 p_decoder_dropout: float=0.1,
                 p_postnet_dropout: float=0.5):
        super().__init__()

        self.frontend = frontend
        std = math.sqrt(2.0 / (self.frontend.vocab_size + d_encoder))
        val = math.sqrt(3.0) * std  # uniform bounds for std
        self.embedding = nn.Embedding(
            self.frontend.vocab_size,
            d_encoder,
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(
                low=-val, high=val)))
        self.encoder = Tacotron2Encoder(d_encoder, encoder_conv_layers,
                                        encoder_kernel_size, p_encoder_dropout)
        self.decoder = Tacotron2Decoder(
            d_mels, reduction_factor, d_encoder, d_prenet, d_attention_rnn,
            d_decoder_rnn, d_attention, attention_filters,
            attention_kernel_size, p_prenet_dropout, p_attention_dropout,
            p_decoder_dropout)
        self.postnet = DecoderPostNet(
            d_mels=d_mels,
            d_hidden=d_postnet,
            kernel_size=postnet_kernel_size,
            padding=int((postnet_kernel_size - 1) / 2),
            num_layers=postnet_conv_layers,
            dropout=p_postnet_dropout)

    def forward(self, text_inputs, mels, text_lens, output_lens=None):
        embedded_inputs = self.embedding(text_inputs)
        encoder_outputs = self.encoder(embedded_inputs, text_lens)

        mask = paddle.tensor.unsqueeze(
            paddle.fluid.layers.sequence_mask(
                x=text_lens, dtype=encoder_outputs.dtype), [-1])
        mel_outputs, stop_logits, alignments = self.decoder(
            encoder_outputs, mels, mask=mask)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        if output_lens is not None:
            mask = paddle.tensor.unsqueeze(
                paddle.fluid.layers.sequence_mask(x=output_lens),
                [-1])  #[B, T, 1]
            mel_outputs = mel_outputs * mask  #[B, T, C]
            mel_outputs_postnet = mel_outputs_postnet * mask  #[B, T, C]
            stop_logits = stop_logits * mask[:, :, 0] + (1 - mask[:, :, 0]
                                                         ) * 1e3  #[B, T]
        outputs = {
            "mel_output": mel_outputs,
            "mel_outputs_postnet": mel_outputs_postnet,
            "stop_logits": stop_logits,
            "alignments": alignments
        }

        return outputs

    @paddle.no_grad()
    def infer(self, text_inputs, stop_threshold=0.5, max_decoder_steps=1000):
        embedded_inputs = self.embedding(text_inputs)
        encoder_outputs = self.encoder(embedded_inputs)
        mel_outputs, stop_logits, alignments = self.decoder.infer(
            encoder_outputs,
            stop_threshold=stop_threshold,
            max_decoder_steps=max_decoder_steps)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = {
            "mel_output": mel_outputs,
            "mel_outputs_postnet": mel_outputs_postnet,
            "stop_logits": stop_logits,
            "alignments": alignments
        }

        return outputs

    @paddle.no_grad()
    def predict(self, text, stop_threshold=0.5, max_decoder_steps=1000):
        # TODO(lifuchen): implement predict function to product mel from texts
        ids = np.asarray(self.frontend(text))
        ids = paddle.unsqueeze(paddle.to_tensor(ids, dtype='int64'), [0])
        outputs = self.infer(ids, stop_threshold, max_decoder_steps)
        return outputs['mel_outputs_postnet'][0].numpy(), outputs[
            'alignments'][0].numpy()

    @classmethod
    def from_pretrained(cls, frontend, config, checkpoint_path):
        model = cls(frontend,
                    d_mels=config.data.d_mels,
                    d_encoder=config.model.d_encoder,
                    encoder_conv_layers=config.model.encoder_conv_layers,
                    encoder_kernel_size=config.model.encoder_kernel_size,
                    d_prenet=config.model.d_prenet,
                    d_attention_rnn=config.model.d_attention_rnn,
                    d_decoder_rnn=config.model.d_decoder_rnn,
                    attention_filters=config.model.attention_filters,
                    attention_kernel_size=config.model.attention_kernel_size,
                    d_attention=config.model.d_attention,
                    d_postnet=config.model.d_postnet,
                    postnet_kernel_size=config.model.postnet_kernel_size,
                    postnet_conv_layers=config.model.postnet_conv_layers,
                    reduction_factor=config.model.reduction_factor,
                    p_encoder_dropout=config.model.p_encoder_dropout,
                    p_prenet_dropout=config.model.p_prenet_dropout,
                    p_attention_dropout=config.model.p_attention_dropout,
                    p_decoder_dropout=config.model.p_decoder_dropout,
                    p_postnet_dropout=config.model.p_postnet_dropout)

        checkpoint.load_parameters(model, checkpoint_path=checkpoint_path)
        return model


class Tacotron2Loss(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, mel_outputs, mel_outputs_postnet, stop_logits,
                mel_targets, stop_tokens):
        mel_loss = paddle.nn.MSELoss()(mel_outputs, mel_targets)
        post_mel_loss = paddle.nn.MSELoss()(mel_outputs_postnet, mel_targets)
        stop_loss = paddle.nn.BCEWithLogitsLoss()(stop_logits, stop_tokens)
        total_loss = mel_loss + post_mel_loss + stop_loss
        losses = dict(
            loss=total_loss,
            mel_loss=mel_loss,
            post_mel_loss=post_mel_loss,
            stop_loss=stop_loss)
        return losses
