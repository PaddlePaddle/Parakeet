# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I
from paddle.fluid.layers import sequence_mask

from parakeet.modules.positioning import position_encoding
from parakeet.modules.attention import (_split_heads, _concat_heads,
                                        scaled_dot_product_attention)
from parakeet.modules import geometry as geo
from parakeet.modules.conv import Conv1dBatchNorm

from typing import Optional


class FastSpeechFeedForwardTransformer(nn.Layer):
    def __init__(self,
                 num_layers,
                 model_dim,
                 num_heads,
                 ffn_dim,
                 ffn_kernel_size,
                 attention_dropout=0.,
                 residual_dropout=0.,
                 num_speakers=1,
                 max_position=1000,
                 input_dim: Optional[int]=None,
                 epsilon=1e-5,
                 scheme="post"):
        super().__init__()
        # optional input layer
        input_dim = input_dim or model_dim
        self.input_dim = input_dim
        self.model_dim = model_dim
        if input_dim != model_dim:
            self.input_fc = nn.Linear(input_dim, model_dim)

        self.pos_embedding = position_encoding(1 + max_position, model_dim)

        self.num_speakers = num_speakers
        if num_speakers > 1:
            self.speaker_embedding = nn.Embedding(num_speakers, model_dim)
            self.speaker_fc = nn.Linear(model_dim, model_dim)

        self.layers = nn.LayerList([
            FastSpeechFFTBlock(model_dim, num_heads, ffn_dim, ffn_kernel_size,
                               attention_dropout, residual_dropout, epsilon,
                               scheme) for _ in range(num_layers)
        ])

    def forward(self, x, mask, speaker_ids=None):
        """
        x: [B, T, C]
        mask: [B, 1, T] or [B, T, T]
        returns: [B, T, C]
        """
        if self.input_dim != self.model_dim:
            x = self.input_fc(x)

        batch_size, time_steps, _ = x.shape
        pos_embed = self.pos_embedding[1:1 + time_steps, :]
        x += pos_embed

        if self.num_speakers > 1:
            speaker_embedding = self.speaker_embedding(speaker_ids)
            speaker_feature = F.softplus(self.speaker_fc(speaker_embedding))
            speaker_feature = paddle.unsqueeze(speaker_feature, 1)  # [B, T, C]
            x += speaker_feature

        for layer in self.layers:
            x, attn = layer(x, mask)
        # we do not return attention here
        return x


class MultiheadAttention(nn.Layer):
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 k_input_dim: Optional[int]=None,
                 v_input_dim: Optional[int]=None,
                 dropout: float=0.):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        depth = model_dim // num_heads
        k_input_dim = k_input_dim or model_dim
        v_input_dim = v_input_dim or model_dim
        self.wq = nn.Linear(model_dim, model_dim)
        self.wk = nn.Linear(k_input_dim, model_dim)
        self.wv = nn.Linear(v_input_dim, model_dim)
        self.wo = nn.Linear(model_dim, model_dim)

        self.num_heads = num_heads
        self.model_dim = model_dim
        self.dropout = dropout

    def forward(self, q, k, v, mask=None):
        q = _split_heads(self.wq(q), self.num_heads)  # (B, h, T, C)
        k = _split_heads(self.wk(k), self.num_heads)
        v = _split_heads(self.wv(v), self.num_heads)
        if mask is not None:
            mask = paddle.unsqueeze(mask, 1)  # unsqueeze for the h dim

        context_vectors, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, dropout=self.dropout, training=self.training)
        context_vectors = _concat_heads(context_vectors)
        context_vectors = self.wo(context_vectors)
        return context_vectors, attention_weights


class FastSpeechSelfAttentionNorm(nn.Layer):
    """Self attention & Layer normalization, both schemes are supported."""

    def __init__(self,
                 model_dim,
                 num_heads,
                 attention_dropout=0.,
                 residual_dropout=0.,
                 epsilon=1e-5,
                 scheme="post"):
        super().__init__()
        if scheme not in ["post", "pre"]:
            raise ValueError("scheme should be 'pre' or 'post'")
        self.scheme = scheme

        self.attention = MultiheadAttention(
            model_dim, num_heads, dropout=attention_dropout)
        self.layer_norm = nn.LayerNorm([model_dim], epsilon=epsilon)
        self.dropout_layer = nn.Dropout(residual_dropout)

    def forward(self, x, mask=None):
        # [B, T, C], [B, 1, T] -> [B, T, C], [B, T, T]
        if self.scheme is "post":
            c, w = self.attention(x, x, x, mask=mask)
            out = self.layer_norm(x + self.dropout_layer(c))
        else:
            normalized_x = self.layer_norm(x)
            c, w = self.attention(
                normalized_x, normalized_x, normalized_x, mask=mask)
            out = x + self.dropout_layer(c)

        c *= paddle.transpose(mask, [0, 2, 1])  # mask padding positions
        return out, w


class FastSpeechFFN(nn.Layer):
    """FFN, it can either be 2 linear or 2 conv1d."""

    def __init__(self, model_dim, hidden_dim, kernel_size=1):
        super().__init__()
        if kernel_size == 1:
            self.layer1 = nn.Linear(model_dim, hidden_dim)
            self.layer2 = nn.Linear(hidden_dim, model_dim)
        else:
            self.layer1 = nn.Conv1D(
                model_dim,
                hidden_dim,
                kernel_size,
                padding="same",
                data_format="NLC")
            self.layer2 = nn.Conv1D(
                hidden_dim,
                model_dim,
                kernel_size,
                padding="same",
                data_format="NLC")

    def forward(self, x, mask=None):
        # [B, T, C], [B, T] -> [B, T, C]
        h = self.layer1(x)
        h = F.relu(h)  # TODO: use mish here?
        h = self.layer2(h)
        h *= paddle.unsqueeze(mask, -1)  # mask padding positions
        return h


class FastSpeechFFNNorm(nn.Layer):
    def __init__(self,
                 model_dim,
                 hidden_dim,
                 kernel_size,
                 residual_dropout=0.,
                 epsilon=1e-5,
                 scheme="post"):
        super().__init__()
        if scheme not in ["post", "pre"]:
            raise ValueError("scheme should be 'pre' or 'post'")
        self.scheme = scheme

        self.ffn = FastSpeechFFN(
            model_dim, hidden_dim, kernel_size=kernel_size)
        self.layer_norm = nn.LayerNorm([model_dim], epsilon=epsilon)
        self.dropout_layer = nn.Dropout(residual_dropout)

    def forward(self, x, mask=None):
        if self.scheme == "post":
            h = self.ffn(x, mask)
            out = self.layer_norm(x + self.dropout_layer(h))
        else:
            normalized_x = self.layer_norm(x)
            h = self.ffn(normalized_x, mask)
            out = x + self.dropout_layer(h)
        out *= paddle.unsqueeze(mask, -1)  # mask padding positions
        return out


class FastSpeechFFTBlock(nn.Layer):
    def __init__(self,
                 model_dim,
                 num_heads,
                 ffn_dim,
                 ffn_kernel_size,
                 attention_dropout=0.,
                 residual_dropout=0.,
                 epsilon=1e-5,
                 scheme="post"):
        super().__init__()
        self.attention = FastSpeechSelfAttentionNorm(
            model_dim, num_heads, attention_dropout, residual_dropout, epsilon,
            scheme)
        self.ffn = FastSpeechFFNNorm(model_dim, ffn_dim, ffn_kernel_size,
                                     residual_dropout, epsilon, scheme)

    def forward(self, x, mask):
        # [B, T, C]
        # [B, 1, T]
        c, w = self.attention(x, mask)
        c = self.ffn(c, paddle.squeeze(mask))
        return c, w


class FastSpeechDurationPredictor(nn.Layer):
    def __init__(self,
                 num_layers: int,
                 input_dim: int,
                 hidden_dim: int,
                 kernel_size: int,
                 dropout: float=0.,
                 epsilon: float=1e-5):
        super().__init__()
        convs = []
        for i in range(num_layers):
            conv = nn.Conv1D(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                kernel_size,
                padding="same",
                data_format="NLC")
            layer_norm = nn.LayerNorm([hidden_dim], epsilon=epsilon)
            act = nn.ReLU6()
            dropout_layer = nn.Dropout(dropout)
            convs.extend([conv, layer_norm, act, dropout_layer])
        self.conv_layers = nn.Sequential(*convs)
        self.output_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        # [B, T, C], [B, T] -> [B, T]
        mask = paddle.unsqueeze(mask, -1)
        x *= mask

        h = self.conv_layers(x)
        h = self.output_fc(h)
        h *= mask
        h = F.relu6(h).squeeze(-1)
        return h


class FastSpeechLengthRegulator(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, durations):
        # [B, T, C], [B, T] -> [B, T', C], [B]
        output_lens = paddle.sum(durations, axis=-1)

        batch_size = x.shape[0]
        expanded_sequences = []
        for i in range(batch_size):
            expanded_sequence = geo.repeat(x[i], durations[i], axis=0)
            expanded_sequences.append(expanded_sequence)
        padded_sequence = geo.pad_sequences(expanded_sequences)
        return padded_sequence, output_lens


class TacotronPostNet(nn.Layer):
    def __init__(self,
                 num_layers,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 dropout=0.,
                 momentum=0.9,
                 epsilon=1e-5):
        super().__init__()
        self.conv_bns = nn.LayerList()
        self.num_layers = num_layers
        for i in range(num_layers):
            convbn = Conv1dBatchNorm(
                input_dim if i == 0 else hidden_dim,
                hidden_dim if i != num_layers - 1 else input_dim,
                kernel_size,
                padding="same",
                data_format="NLC",
                momentum=momentum,
                epsilon=epsilon)
            self.conv_bns.append(convbn)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, mask):
        # [B, T, C], [B, T] -> [B, T, C]
        mask = paddle.unsqueeze(mask, -1)
        for i, convbn in enumerate(self.conv_bns):
            x = convbn(x)
            if i != self.num_layers - 1:
                x = paddle.tanh(x)
            x = self.dropout_layer(x)
        x *= mask
        return x


class FastSpeechVariancePredictor(nn.Layer):
    def __init__(self,
                 num_layers: int,
                 input_dim: int,
                 hidden_dim: int,
                 kernel_size: int,
                 num_speakers: int=1,
                 speaker_embedding_size: Optional[int]=None,
                 dropout: float=0.,
                 epsilon: float=1e-5):
        super().__init__()
        convs = []
        for i in range(num_layers):
            conv = nn.Conv1D(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                kernel_size,
                padding="same",
                data_format="NLC")
            act = nn.ReLU()
            ln = nn.LayerNorm([hidden_dim], epsilon=epsilon)
            dropout_layer = nn.Dropout(dropout)
            convs.extend([conv, act, ln, dropout_layer])
        self.conv_layers = nn.Sequential(*convs)
        self.output_fc = nn.Linear(hidden_dim, 1)

        self.num_speakers = num_speakers
        if num_speakers > 1:
            self.speaker_embedding = nn.Embedding(num_speakers,
                                                  speaker_embedding_size)
            self.speaker_fc = nn.Linear(speaker_embedding_size, input_dim)

    def forward(self, x, speaker_ids, mask):
        # [B, T, C], [B], [B, T] -> [B, T]
        if self.num_speakers > 1:
            speaker_embed = self.speaker_embeddings(speaker_ids)
            speaker_features = F.softplus(self.speaker_fc(speaker_embed))
            x += paddle.unsqueeze(speaker_features, 1)

        x *= paddle.unsqueeze(mask, -1)

        h = self.conv_layers(x)
        out = self.output_fc(h)
        out = paddle.squeeze(-1) * mask
        return out


class FastSpeech(nn.Layer):
    def __init__(
            self,
            vocab_size,
            num_speakers,
            # encoder params
            encoder_num_layers,
            encoder_dim,
            encoder_num_heads,
            encoder_max_position,
            encoder_ffn_dim,
            encoder_ffn_kernel_size,
            # decoder params
            decoder_num_layers,
            decoder_dim,
            decoder_num_heads,
            decoder_max_position,
            decoder_ffn_dim,
            decoder_ffn_kernel_size,
            # encoder & decoder common
            attention_dropout,
            residual_dropout,
            # duration predictor
            duration_predictor_num_layers,
            duration_predictor_dim,
            duration_predictor_kernel_size,
            duration_predictor_dropout,
            # output
            mel_dim,
            # postnet
            postnet_num_layers,
            postnet_dim,
            postnet_kernel_size,
            postnet_dropout,
            # other
            padding_idx=0,
            momentum=0.9,
            epsilon=1e-5,
            scheme="post"):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, encoder_dim, padding_idx=padding_idx)
        self.encoder = FastSpeechFeedForwardTransformer(
            encoder_num_layers,
            encoder_dim,
            encoder_num_heads,
            encoder_ffn_dim,
            encoder_ffn_kernel_size,
            attention_dropout,
            residual_dropout,
            num_speakers=num_speakers,
            max_position=encoder_max_position,
            epsilon=epsilon,
            scheme=scheme)
        self.duration_predictor = FastSpeechDurationPredictor(
            duration_predictor_num_layers,
            encoder_dim,
            duration_predictor_dim,
            duration_predictor_kernel_size,
            duration_predictor_dropout,
            epsilon=epsilon)
        self.length_regulator = FastSpeechLengthRegulator()
        self.decoder = FastSpeechFeedForwardTransformer(
            decoder_num_layers,
            decoder_dim,
            decoder_num_heads,
            decoder_ffn_dim,
            decoder_ffn_kernel_size,
            attention_dropout,
            residual_dropout,
            num_speakers=num_speakers,
            max_position=decoder_max_position,
            input_dim=encoder_dim,
            epsilon=epsilon,
            scheme=scheme)
        self.mel_output_fc = nn.Linear(decoder_dim, mel_dim)
        self.postnet = TacotronPostNet(
            postnet_num_layers,
            mel_dim,
            postnet_dim,
            postnet_kernel_size,
            postnet_dropout,
            momentum=momentum,
            epsilon=epsilon)

    def forward(self, text_ids, speaker_ids, durations, text_lens):
        dtype = paddle.get_default_dtype()
        encoder_padding_mask = sequence_mask(text_lens, dtype=dtype)
        encoder_attention_mask = encoder_padding_mask.unsqueeze(1)

        embedding = self.embedding(text_ids)
        encoder_output = self.encoder(embedding, encoder_attention_mask,
                                      speaker_ids)

        # detach the gradient of duration predictor
        # a difference here
        predicted_durations = self.duration_predictor(encoder_output.detach(),
                                                      encoder_padding_mask)

        expanded_outputs, mel_lens = self.length_regulator(encoder_output,
                                                           durations)
        decoder_padding_mask = sequence_mask(mel_lens, dtype=dtype)
        decoder_attention_mask = decoder_padding_mask.unsqueeze(1)

        decoder_ouputs = self.decoder(
            expanded_outputs,
            decoder_attention_mask,
            speaker_ids, )
        decoder_mel = self.mel_output_fc(decoder_ouputs)
        postnet_mel = decoder_mel + self.postnet(decoder_mel,
                                                 decoder_padding_mask)

        return decoder_mel, postnet_mel, predicted_durations

    def inference(self, text_ids, speaker_ids, text_lens, speed_ratios):
        dtype = paddle.get_default_dtype()
        encoder_padding_mask = sequence_mask(text_lens, dtype=dtype)
        encoder_attention_mask = encoder_padding_mask.unsqueeze(1)

        embedding = self.embedding(text_ids)
        encoder_output = self.encoder(embedding, encoder_attention_mask,
                                      speaker_ids)

        # detach the gradient flow of duration predictor
        # a difference here
        predicted_log_durations = self.duration_predictor(
            encoder_output.detach(), encoder_padding_mask)
        predicted_durations = paddle.exp(predicted_log_durations) - 1.

        if speed_ratios is None:
            speed_ratios = paddle.ones([1], dtype=dtype)
        speed_ratios = paddle.unsqueeze(speed_ratios, -1)
        predicted_durations = paddle.round(predicted_durations *
                                           speed_ratios).astype("int32")

        expanded_outputs, mel_lens = self.length_regulator(encoder_output,
                                                           predicted_durations)
        decoder_padding_mask = sequence_mask(mel_lens, dtype=dtype)
        decoder_attention_mask = decoder_padding_mask.unsqueeze(1)

        decoder_ouputs = self.decoder(expanded_outputs, decoder_attention_mask,
                                      speaker_ids)
        decoder_mel = self.mel_output_fc(decoder_ouputs)
        postnet_mel = decoder_mel + self.postnet(decoder_mel,
                                                 decoder_padding_mask)

        return decoder_mel, postnet_mel, predicted_durations


# TODO: implement FastSpeech2
class FastSpeech2(nn.Layer):
    def __init__(
            self,
            vocab_size,
            num_speakers,
            # encoder params
            encoder_num_layers,
            encoder_dim,
            encoder_num_heads,
            encoder_max_position,
            encoder_ffn_dim,
            encoder_ffn_kernel_size,
            # decoder params
            decoder_num_layers,
            decoder_dim,
            decoder_num_heads,
            decoder_max_position,
            decoder_ffn_dim,
            decoder_ffn_kernel_size,
            # encoder & decoder common
            attention_dropout,
            residual_dropout,
            # duration predictor
            duration_predictor_num_layers,
            duration_predictor_dim,
            duration_predictor_kernel_size,
            duration_predictor_dropout,
            # output
            mel_dim,
            # postnet
            postnet_num_layers,
            postnet_dim,
            postnet_kernel_size,
            postnet_dropout,
            # variance predictor
            variance_predictor_num_layers,
            variance_predictor_dim,
            variance_predictor_kernel_size,
            variance_predictor_dropout,
            # other
            padding_idx=0,
            momentum=0.9,
            epsilon=1e-5,
            scheme="post"):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, encoder_dim, padding_idx=padding_idx)
        self.encoder = FastSpeechFeedForwardTransformer(
            encoder_num_layers,
            encoder_dim,
            encoder_num_heads,
            encoder_ffn_dim,
            encoder_ffn_kernel_size,
            attention_dropout,
            residual_dropout,
            num_speakers=num_speakers,
            max_position=encoder_max_position,
            epsilon=epsilon,
            scheme=scheme)
        self.duration_predictor = FastSpeechDurationPredictor(
            duration_predictor_num_layers,
            encoder_dim,
            duration_predictor_dim,
            duration_predictor_kernel_size,
            duration_predictor_dropout,
            epsilon=epsilon)
        self.length_regulator = FastSpeechLengthRegulator()
        self.decoder = FastSpeechFeedForwardTransformer(
            decoder_num_layers,
            decoder_dim,
            decoder_num_heads,
            decoder_ffn_dim,
            decoder_ffn_kernel_size,
            attention_dropout,
            residual_dropout,
            num_speakers=num_speakers,
            max_position=decoder_max_position,
            input_dim=encoder_dim,
            epsilon=epsilon,
            scheme=scheme)
        self.mel_output_fc = nn.Linear(decoder_dim, mel_dim)
        self.postnet = TacotronPostNet(
            postnet_num_layers,
            mel_dim,
            postnet_dim,
            postnet_kernel_size,
            postnet_dropout,
            momentum=momentum,
            epsilon=epsilon)
        # difference here?
        self.f0_predictor = FastSpeechVariancePredictor(
            variance_predictor_num_layers,
            embed_dim,
            variance_predictor_dim,
            variancce_predictor_kernel_size,
            num_speakers,
            speaker_embedding_size=embed_dim)
        self.energy_predictor = FastSpeechVariancePredictor(
            variance_predictor_num_layers,
            embed_dim,
            variance_predictor_dim,
            variancce_predictor_kernel_size,
            num_speakers,
            speaker_embedding_size=embed_dim)
        #self.duration_predictor = FastSpeechVariancePredictor(
        #variance_predictor_num_layers,
        #embed_dim,
        #variance_predictor_dim,
        #variancce_predictor_kernel_size,
        #num_speakers,
        #speaker_embedding_size=embed_dim)
        self.f0_embedding = nn.Conv1D(
            1, encoder_dim, kernel_size=9, padding="same", data_format="NLC")
        self.f0_dropout_layer = nn.Dropout(0.5)
        self.energy_embeddings = nn.Conv1D(
            1, encoder_dim, kernel_size=9, padding="same", data_format="NLC")
        self.energy_dropout = nn.Dropout(0.5)

    def forward(self, text_ids, speaker_ids, durations, text_lens):
        dtype = paddle.get_default_dtype()
        encoder_padding_mask = sequence_mask(text_lens, dtype=dtype)
        encoder_attention_mask = encoder_padding_mask.unsqueeze(1)

        embedding = self.embedding(text_ids)
        encoder_output = self.encoder(embedding, encoder_attention_mask,
                                      speaker_ids)

        # detach the gradient of duration predictor
        # a difference here
        predicted_durations = self.duration_predictor(encoder_output.detach(),
                                                      encoder_padding_mask)

        expanded_outputs, mel_lens = self.length_regulator(encoder_output,
                                                           durations)
        decoder_padding_mask = sequence_mask(mel_lens, dtype=dtype)
        decoder_attention_mask = decoder_padding_mask.unsqueeze(1)

        decoder_ouputs = self.decoder(
            expanded_outputs,
            decoder_attention_mask,
            speaker_ids, )
        decoder_mel = self.mel_output_fc(decoder_ouputs)
        postnet_mel = decoder_mel + self.postnet(decoder_mel,
                                                 decoder_padding_mask)

        return decoder_mel, postnet_mel, predicted_durations

    def inference(self, text_ids, speaker_ids, text_lens, speed_ratios):
        dtype = paddle.get_default_dtype()
        encoder_padding_mask = sequence_mask(text_lens, dtype=dtype)
        encoder_attention_mask = encoder_padding_mask.unsqueeze(1)

        embedding = self.embedding(text_ids)
        encoder_output = self.encoder(embedding, encoder_attention_mask,
                                      speaker_ids)

        # detach the gradient flow of duration predictor
        # a difference here
        predicted_log_durations = self.duration_predictor(
            encoder_output.detach(), encoder_padding_mask)
        predicted_durations = paddle.exp(predicted_log_durations) - 1.

        if speed_ratios is None:
            speed_ratios = paddle.ones([1], dtype=dtype)
        speed_ratios = paddle.unsqueeze(speed_ratios, -1)
        predicted_durations = paddle.round(predicted_durations *
                                           speed_ratios).astype("int32")

        expanded_outputs, mel_lens = self.length_regulator(encoder_output,
                                                           predicted_durations)
        decoder_padding_mask = sequence_mask(mel_lens, dtype=dtype)
        decoder_attention_mask = decoder_padding_mask.unsqueeze(1)

        decoder_ouputs = self.decoder(expanded_outputs, decoder_attention_mask,
                                      speaker_ids)
        decoder_mel = self.mel_output_fc(decoder_ouputs)
        postnet_mel = decoder_mel + self.postnet(decoder_mel,
                                                 decoder_padding_mask)

        return decoder_mel, postnet_mel, predicted_durations
