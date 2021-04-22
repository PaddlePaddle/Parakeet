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
from tqdm import trange
from parakeet.models.tacotron2 import DecoderPreNet, DecoderPostNet, Tacotron2Encoder, Tacotron2Decoder, Tacotron2Loss

__all__ = ["Tacotron2", "Tacotron2Loss"]


class Tacotron2Decoder(nn.Layer):
    """Tacotron2 decoder module for Tacotron2.

    Parameters
    ----------
    d_mels: int
        The number of mel bands.

    reduction_factor: int
        The reduction factor of tacotron.
    
    d_encoder: int
        The hidden size of encoder.

    d_prenet: int
        The hidden size in decoder prenet.

    d_attention_rnn: int
        The attention rnn layer hidden size.

    d_decoder_rnn: int
        The decoder rnn layer hidden size.
    
    d_attention: int
        The hidden size of the linear layer in location sensitive attention.

    attention_filters: int
        The filter size of the conv layer in location sensitive attention.
            
    attention_kernel_size: int
        The kernel size of the conv layer in location sensitive attention.

    p_prenet_dropout: float
        The droput probability in decoder prenet.

    p_attention_dropout: float
        The droput probability in location sensitive attention.

    p_decoder_dropout: float
        The droput probability in decoder.
    """

    def __init__(self, d_mels: int, reduction_factor: int, d_encoder: int,
                 d_prenet: int, d_attention_rnn: int, d_decoder_rnn: int,
                 d_attention: int, attention_filters: int,
                 attention_kernel_size: int, p_prenet_dropout: float,
                 p_attention_dropout: float, p_decoder_dropout: float):
        super().__init__()
        self.d_mels = d_mels
        self.reduction_factor = reduction_factor
        self.d_encoder = d_encoder
        self.d_attention_rnn = d_attention_rnn
        self.d_decoder_rnn = d_decoder_rnn
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout

        self.prenet = DecoderPreNet(d_mels * reduction_factor,
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
        """init states be used in decoder
        """
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

        self.attention_weights = paddle.zeros(shape=[batch_size, MAX_TIME],
                                              dtype=key.dtype)
        self.attention_weights_cum = paddle.zeros(shape=[batch_size, MAX_TIME],
                                                  dtype=key.dtype)
        self.attention_context = paddle.zeros(
            shape=[batch_size, self.d_encoder], dtype=key.dtype)

        self.key = key  # [B, T, C]
        self.processed_key = self.attention_layer.key_layer(key)  # [B, T, C]

    def _decode(self, query):
        """decode one time step
        """
        cell_input = paddle.concat([query, self.attention_context], axis=-1)

        # The first lstm layer
        _, (self.attention_hidden, self.attention_cell) = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(self.attention_hidden,
                                          self.p_attention_dropout,
                                          training=self.training)

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
        self.decoder_hidden = F.dropout(self.decoder_hidden,
                                        p=self.p_decoder_dropout,
                                        training=self.training)

        # decode output one step
        decoder_hidden_attention_context = paddle.concat(
            [self.decoder_hidden, self.attention_context], axis=-1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)
        stop_logit = self.stop_layer(decoder_hidden_attention_context)
        return decoder_output, stop_logit, self.attention_weights

    def forward(self, keys, querys, mask):
        """Calculate forward propagation of tacotron2 decoder.

        Parameters
        ----------
        keys: Tensor[shape=(B, T_key, C)]
            Batch of the sequences of padded output from encoder.
        
        querys: Tensor[shape(B, T_query, C)]
            Batch of the sequences of padded mel spectrogram.
        
        mask: Tensor
            Mask generated with text length. Shape should be (B, T_key, T_query) or broadcastable shape.
        
        Returns
        -------
        mel_output: Tensor [shape=(B, T_query, C)]
            Output sequence of features.

        stop_logits: Tensor [shape=(B, T_query)]
            Output sequence of stop logits.

        alignments: Tensor [shape=(B, T_query, T_key)]
            Attention weights.
        """
        querys = paddle.reshape(
            querys,
            [querys.shape[0], querys.shape[1] // self.reduction_factor, -1])
        querys = paddle.concat([
            paddle.zeros(shape=[querys.shape[0], 1, querys.shape[-1]],
                         dtype=querys.dtype), querys
        ],
            axis=1)
        querys = self.prenet(querys)

        self._initialize_decoder_states(keys)
        self.mask = mask

        mel_outputs, stop_logits, alignments = [], [], []
        while len(mel_outputs
                  ) < querys.shape[1] - 1:  # Ignore the last time step
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
        """Calculate forward propagation of tacotron2 decoder.

        Parameters
        ----------
        keys: Tensor [shape=(B, T_key, C)]
            Batch of the sequences of padded output from encoder.
        
        stop_threshold: float, optional
            Stop synthesize when stop logit is greater than this stop threshold. Defaults to 0.5.
        
        max_decoder_steps: int, optional
            Number of max step when synthesize. Defaults to 1000.
        
        Returns
        -------
        mel_output: Tensor [shape=(B, T_mel, C)]
            Output sequence of features.

        stop_logits: Tensor [shape=(B, T_mel)]
            Output sequence of stop logits.

        alignments: Tensor [shape=(B, T_mel, T_key)]
            Attention weights.

        """
        query = paddle.zeros(
            shape=[key.shape[0], self.d_mels * self.reduction_factor],
            dtype=key.dtype)  # [B, C]

        self._initialize_decoder_states(key)
        T_enc = key.shape[1]
        self.mask = None
        first_hit_end = None

        mel_outputs, stop_logits, alignments = [], [], []
        for i in trange(max_decoder_steps):
            query = self.prenet(query)
            mel_output, stop_logit, alignment = self._decode(query)

            mel_outputs += [mel_output]
            stop_logits += [stop_logit]
            alignments += [alignment]

            if F.sigmoid(stop_logit) > stop_threshold:
                print("hits stop condition!")
                break
            if int(paddle.argmax(alignment[0])) == T_enc - 1:
                if (first_hit_end is None):
                    first_hit_end = i
            if first_hit_end is not None and i > (first_hit_end + 10):
                print("content exhausted!")
                break
            if len(mel_outputs) == max_decoder_steps:
                print("Warning! Reached max decoder steps!!!")
                break

            query = mel_output

        alignments = paddle.stack(alignments, axis=1)
        stop_logits = paddle.concat(stop_logits, axis=1)
        mel_outputs = paddle.stack(mel_outputs, axis=1)

        return mel_outputs, stop_logits, alignments


class Tacotron2(nn.Layer):
    """Tacotron2 model for end-to-end text-to-speech (E2E-TTS).

    This is a model of Spectrogram prediction network in Tacotron2 described
    in `Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions 
    <https://arxiv.org/abs/1712.05884>`_,
    which converts the sequence of characters
    into the sequence of mel spectrogram.

    Parameters
    ----------
    frontend : parakeet.frontend.Phonetics
        Frontend used to preprocess text.

    d_mels: int
        Number of mel bands.
    
    d_encoder: int
        Hidden size in encoder module.
    
    encoder_conv_layers: int
        Number of conv layers in encoder.

    encoder_kernel_size: int
        Kernel size of conv layers in encoder.

    d_prenet: int
        Hidden size in decoder prenet.

    d_attention_rnn: int
        Attention rnn layer hidden size in decoder.

    d_decoder_rnn: int
        Decoder rnn layer hidden size in decoder.

    attention_filters: int
        Filter size of the conv layer in location sensitive attention.
            
    attention_kernel_size: int
        Kernel size of the conv layer in location sensitive attention.

    d_attention: int
        Hidden size of the linear layer in location sensitive attention.

    d_postnet: int
        Hidden size of postnet.

    postnet_kernel_size: int
        Kernel size of the conv layer in postnet.

    postnet_conv_layers: int
        Number of conv layers in postnet.

    reduction_factor: int
        Reduction factor of tacotron2.

    p_encoder_dropout: float
        Droput probability in encoder.

    p_prenet_dropout: float
        Droput probability in decoder prenet.

    p_attention_dropout: float
        Droput probability in location sensitive attention.

    p_decoder_dropout: float
        Droput probability in decoder.

    p_postnet_dropout: float
        Droput probability in postnet.

    """

    def __init__(self,
                 frontend: parakeet.frontend.Phonetics,
                 d_mels: int = 80,
                 d_encoder: int = 512,
                 encoder_conv_layers: int = 3,
                 encoder_kernel_size: int = 5,
                 d_prenet: int = 256,
                 d_attention_rnn: int = 1024,
                 d_decoder_rnn: int = 1024,
                 attention_filters: int = 32,
                 attention_kernel_size: int = 31,
                 d_attention: int = 128,
                 d_postnet: int = 512,
                 postnet_kernel_size: int = 5,
                 postnet_conv_layers: int = 5,
                 reduction_factor: int = 1,
                 p_encoder_dropout: float = 0.5,
                 p_prenet_dropout: float = 0.5,
                 p_attention_dropout: float = 0.1,
                 p_decoder_dropout: float = 0.1,
                 p_postnet_dropout: float = 0.5,
                 n_tones=None,
                 speaker_embed_dim=None):
        super().__init__()

        self.frontend = frontend
        std = math.sqrt(2.0 / (self.frontend.vocab_size + d_encoder))
        val = math.sqrt(3.0) * std  # uniform bounds for std
        self.embedding = nn.Embedding(
            self.frontend.vocab_size,
            d_encoder,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.Uniform(low=-val, high=val)))
        if n_tones:
            self.embedding_tones = nn.Embedding(
                n_tones,
                d_encoder,
                padding_idx=0,
                weight_attr=paddle.ParamAttr(
                    initializer=nn.initializer.Uniform(low=-0.1 * val,
                                                       high=0.1 * val)))
        self.toned = n_tones is not None
        self.encoder = Tacotron2Encoder(d_encoder, encoder_conv_layers,
                                        encoder_kernel_size, p_encoder_dropout)
        if speaker_embed_dim:
            d_encoder += speaker_embed_dim
        self.decoder = Tacotron2Decoder(
            d_mels, reduction_factor, d_encoder, d_prenet, d_attention_rnn,
            d_decoder_rnn, d_attention, attention_filters,
            attention_kernel_size, p_prenet_dropout, p_attention_dropout,
            p_decoder_dropout)
        self.postnet = DecoderPostNet(d_mels=d_mels * reduction_factor,
                                      d_hidden=d_postnet,
                                      kernel_size=postnet_kernel_size,
                                      num_layers=postnet_conv_layers,
                                      dropout=p_postnet_dropout)

    def forward(self,
                text_inputs,
                mels,
                text_lens,
                output_lens=None,
                tones=None,
                utterance_embeds=None):
        """Calculate forward propagation of tacotron2.

        Parameters
        ----------
        text_inputs: Tensor [shape=(B, T_text)]
            Batch of the sequencees of padded character ids.
        
        mels: Tensor [shape(B, T_mel, C)]
            Batch of the sequences of padded mel spectrogram.
        
        text_lens: Tensor [shape=(B,)]
            Batch of lengths of each text input batch.
        
        output_lens: Tensor [shape=(B,)], optional
            Batch of lengths of each mels batch. Defaults to None.
        
        Returns
        -------
        outputs : Dict[str, Tensor]
            
            mel_output: output sequence of features (B, T_mel, C);

            mel_outputs_postnet: output sequence of features after postnet (B, T_mel, C);

            stop_logits: output sequence of stop logits (B, T_mel);

            alignments: attention weights (B, T_mel, T_text).
        """
        embedded_inputs = self.embedding(text_inputs)
        if self.toned:
            embedded_inputs += self.embedding_tones(tones)
            # embedded_inputs = paddle.concat([embedded_inputs, self.embedding_tones(tones)], -1)
        encoder_outputs = self.encoder(embedded_inputs, text_lens)
        if utterance_embeds is not None:
            utterance_embeds = paddle.unsqueeze(utterance_embeds, 1)
            utterance_embeds = paddle.expand(
                utterance_embeds, [-1, encoder_outputs.shape[1], -1])
            encoder_outputs = paddle.concat(
                [encoder_outputs, utterance_embeds], -1)

        mask = paddle.tensor.unsqueeze(
            paddle.fluid.layers.sequence_mask(x=text_lens,
                                              dtype=encoder_outputs.dtype),
            [-1])
        mel_outputs, stop_logits, alignments = self.decoder(encoder_outputs,
                                                            mels,
                                                            mask=mask)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        if output_lens is not None:
            mask = paddle.tensor.unsqueeze(
                paddle.fluid.layers.sequence_mask(x=output_lens),
                [-1])  # [B, T, 1]
            mel_outputs = mel_outputs * mask  # [B, T, C]
            mel_outputs_postnet = mel_outputs_postnet * mask  # [B, T, C]
            stop_logits = stop_logits * mask[:, :, 0] + (
                1 - mask[:, :, 0]) * 1e3  # [B, T]
        outputs = {
            "mel_output": mel_outputs,
            "mel_outputs_postnet": mel_outputs_postnet,
            "stop_logits": stop_logits,
            "alignments": alignments
        }

        return outputs

    @paddle.no_grad()
    def infer(self,
              text_inputs,
              stop_threshold=0.5,
              max_decoder_steps=1000,
              tones=None,
              utterance_embeds=None):
        """Generate the mel sepctrogram of features given the sequences of character ids.

        Parameters
        ----------
        text_inputs: Tensor [shape=(B, T_text)]
            Batch of the sequencees of padded character ids.
        
        stop_threshold: float, optional
            Stop synthesize when stop logit is greater than this stop threshold. Defaults to 0.5.
        
        max_decoder_steps: int, optional
            Number of max step when synthesize. Defaults to 1000.
        
        Returns
        -------
        outputs : Dict[str, Tensor]

            mel_output: output sequence of sepctrogram (B, T_mel, C);

            mel_outputs_postnet: output sequence of sepctrogram after postnet (B, T_mel, C);

            stop_logits: output sequence of stop logits (B, T_mel);

            alignments: attention weights (B, T_mel, T_text).
        """
        embedded_inputs = self.embedding(text_inputs)
        if self.toned:
            embedded_inputs += self.embedding_tones(tones)
        encoder_outputs = self.encoder(embedded_inputs)

        if utterance_embeds is not None:
            utterance_embeds = paddle.unsqueeze(utterance_embeds, 1)
            utterance_embeds = paddle.expand(
                utterance_embeds, [-1, encoder_outputs.shape[1], -1])
            encoder_outputs = paddle.concat(
                [encoder_outputs, utterance_embeds], -1)

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
