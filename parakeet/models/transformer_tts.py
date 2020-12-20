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
from tqdm import trange
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I

import parakeet
from parakeet.modules.attention import _split_heads, _concat_heads, drop_head, scaled_dot_product_attention
from parakeet.modules.transformer import PositionwiseFFN
from parakeet.modules import masking
from parakeet.modules.conv import Conv1dBatchNorm
from parakeet.modules import positional_encoding as pe
from parakeet.modules import losses as L
from parakeet.utils import checkpoint, scheduler

__all__ = ["TransformerTTS", "TransformerTTSLoss"]


# Transformer TTS's own implementation of transformer
class MultiheadAttention(nn.Layer):
    """Multihead scaled dot product attention with drop head. See 
    [Scheduled DropHead: A Regularization Method for Transformer Models](https://arxiv.org/abs/2004.13342) 
    for details.
    
    Another deviation is that it concats the input query and context vector before
    applying the output projection.
    """

    def __init__(self,
                 model_dim,
                 num_heads,
                 k_dim=None,
                 v_dim=None,
                 k_input_dim=None,
                 v_input_dim=None):
        """
        Args:
            model_dim (int): the feature size of query.
            num_heads (int): the number of attention heads.
            k_dim (int, optional): feature size of the key of each scaled dot 
                product attention. If not provided, it is set to 
                model_dim / num_heads. Defaults to None.
            v_dim (int, optional): feature size of the key of each scaled dot 
                product attention. If not provided, it is set to 
                model_dim / num_heads. Defaults to None.

        Raises:
            ValueError: if model_dim is not divisible by num_heads
        """
        super(MultiheadAttention, self).__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        depth = model_dim // num_heads
        k_dim = k_dim or depth
        v_dim = v_dim or depth
        k_input_dim = k_input_dim or model_dim
        v_input_dim = v_input_dim or model_dim
        self.affine_q = nn.Linear(model_dim, num_heads * k_dim)
        self.affine_k = nn.Linear(k_input_dim, num_heads * k_dim)
        self.affine_v = nn.Linear(v_input_dim, num_heads * v_dim)
        self.affine_o = nn.Linear(model_dim + num_heads * v_dim, model_dim)

        self.num_heads = num_heads
        self.model_dim = model_dim

    def forward(self, q, k, v, mask, drop_n_heads=0):
        """
        Compute context vector and attention weights.
        
        Args:
            q (Tensor): shape(batch_size, time_steps_q, model_dim), the queries.
            k (Tensor): shape(batch_size, time_steps_k, model_dim), the keys.
            v (Tensor): shape(batch_size, time_steps_k, model_dim), the values.
            mask (Tensor): shape(batch_size, times_steps_q, time_steps_k) or 
                broadcastable shape, dtype: float32 or float64, the mask.

        Returns:
            out (Tensor), shape(batch_size, time_steps_q, model_dim), the context vector.
            attention_weights (Tensor): shape(batch_size, times_steps_q, time_steps_k), the attention weights.
        """
        q_in = q
        q = _split_heads(self.affine_q(q), self.num_heads)  # (B, h, T, C)
        k = _split_heads(self.affine_k(k), self.num_heads)
        v = _split_heads(self.affine_v(v), self.num_heads)
        if mask is not None:
            mask = paddle.unsqueeze(mask, 1)  # unsqueeze for the h dim

        context_vectors, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, training=self.training)
        context_vectors = drop_head(context_vectors, drop_n_heads,
                                    self.training)
        context_vectors = _concat_heads(context_vectors)  # (B, T, h*C)

        concat_feature = paddle.concat([q_in, context_vectors], -1)
        out = self.affine_o(concat_feature)
        return out, attention_weights


class TransformerEncoderLayer(nn.Layer):
    """
    Transformer encoder layer.
    """

    def __init__(self, d_model, n_heads, d_ffn, dropout=0.):
        """
        Args:
            d_model (int): the feature size of the input, and the output.
            n_heads (int): the number of heads in the internal MultiHeadAttention layer.
            d_ffn (int): the hidden size of the internal PositionwiseFFN.
            dropout (float, optional): the probability of the dropout in 
                MultiHeadAttention and PositionwiseFFN. Defaults to 0.
        """
        super(TransformerEncoderLayer, self).__init__()
        self.self_mha = MultiheadAttention(d_model, n_heads)
        self.layer_norm1 = nn.LayerNorm([d_model], epsilon=1e-6)

        self.ffn = PositionwiseFFN(d_model, d_ffn, dropout)
        self.layer_norm2 = nn.LayerNorm([d_model], epsilon=1e-6)

        self.dropout = dropout

    def _forward_mha(self, x, mask, drop_n_heads):
        # PreLN scheme: Norm -> SubLayer -> Dropout -> Residual
        x_in = x
        x = self.layer_norm1(x)
        context_vector, attn_weights = self.self_mha(x, x, x, mask,
                                                     drop_n_heads)
        context_vector = x_in + F.dropout(
            context_vector, self.dropout, training=self.training)
        return context_vector, attn_weights

    def _forward_ffn(self, x):
        # PreLN scheme: Norm -> SubLayer -> Dropout -> Residual
        x_in = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        out = x_in + F.dropout(x, self.dropout, training=self.training)
        return out

    def forward(self, x, mask, drop_n_heads=0):
        """
        Args:
            x (Tensor): shape(batch_size, time_steps, d_model), the decoder input.
            mask (Tensor): shape(batch_size, 1, time_steps), the padding mask.
        
        Returns:
            x (Tensor): shape(batch_size, time_steps, d_model), the decoded.
            attn_weights (Tensor), shape(batch_size, n_heads, time_steps, time_steps), self attention.
        """
        x, attn_weights = self._forward_mha(x, mask, drop_n_heads)
        x = self._forward_ffn(x)
        return x, attn_weights


class TransformerDecoderLayer(nn.Layer):
    """
    Transformer decoder layer.
    """

    def __init__(self, d_model, n_heads, d_ffn, dropout=0., d_encoder=None):
        """
        Args:
            d_model (int): the feature size of the input, and the output.
            n_heads (int): the number of heads in the internal MultiHeadAttention layer.
            d_ffn (int): the hidden size of the internal PositionwiseFFN.
            dropout (float, optional): the probability of the dropout in 
                MultiHeadAttention and PositionwiseFFN. Defaults to 0.
        """
        super(TransformerDecoderLayer, self).__init__()
        self.self_mha = MultiheadAttention(d_model, n_heads)
        self.layer_norm1 = nn.LayerNorm([d_model], epsilon=1e-6)

        self.cross_mha = MultiheadAttention(
            d_model, n_heads, k_input_dim=d_encoder, v_input_dim=d_encoder)
        self.layer_norm2 = nn.LayerNorm([d_model], epsilon=1e-6)

        self.ffn = PositionwiseFFN(d_model, d_ffn, dropout)
        self.layer_norm3 = nn.LayerNorm([d_model], epsilon=1e-6)

        self.dropout = dropout

    def _forward_self_mha(self, x, mask, drop_n_heads):
        # PreLN scheme: Norm -> SubLayer -> Dropout -> Residual
        x_in = x
        x = self.layer_norm1(x)
        context_vector, attn_weights = self.self_mha(x, x, x, mask,
                                                     drop_n_heads)
        context_vector = x_in + F.dropout(
            context_vector, self.dropout, training=self.training)
        return context_vector, attn_weights

    def _forward_cross_mha(self, q, k, v, mask, drop_n_heads):
        # PreLN scheme: Norm -> SubLayer -> Dropout -> Residual
        q_in = q
        q = self.layer_norm2(q)
        context_vector, attn_weights = self.cross_mha(q, k, v, mask,
                                                      drop_n_heads)
        context_vector = q_in + F.dropout(
            context_vector, self.dropout, training=self.training)
        return context_vector, attn_weights

    def _forward_ffn(self, x):
        # PreLN scheme: Norm -> SubLayer -> Dropout -> Residual
        x_in = x
        x = self.layer_norm3(x)
        x = self.ffn(x)
        out = x_in + F.dropout(x, self.dropout, training=self.training)
        return out

    def forward(self, q, k, v, encoder_mask, decoder_mask, drop_n_heads=0):
        """
        Args:
            q (Tensor): shape(batch_size, time_steps_q, d_model), the decoder input.
            k (Tensor): shape(batch_size, time_steps_k, d_model), keys.
            v (Tensor): shape(batch_size, time_steps_k, d_model), values
            encoder_mask (Tensor): shape(batch_size, 1, time_steps_k) encoder padding mask.
            decoder_mask (Tensor): shape(batch_size, time_steps_q, time_steps_q) or broadcastable shape, decoder padding mask.
        
        Returns:
            q (Tensor): shape(batch_size, time_steps_q, d_model), the decoded.
            self_attn_weights (Tensor), shape(batch_size, n_heads, time_steps_q, time_steps_q), decoder self attention.
            cross_attn_weights (Tensor), shape(batch_size, n_heads, time_steps_q, time_steps_k), decoder-encoder cross attention.
        """
        q, self_attn_weights = self._forward_self_mha(q, decoder_mask,
                                                      drop_n_heads)
        q, cross_attn_weights = self._forward_cross_mha(q, k, v, encoder_mask,
                                                        drop_n_heads)
        q = self._forward_ffn(q)
        return q, self_attn_weights, cross_attn_weights


class TransformerEncoder(nn.LayerList):
    def __init__(self, d_model, n_heads, d_ffn, n_layers, dropout=0.):
        super(TransformerEncoder, self).__init__()
        for _ in range(n_layers):
            self.append(
                TransformerEncoderLayer(d_model, n_heads, d_ffn, dropout))

    def forward(self, x, mask, drop_n_heads=0):
        """
        Args:
            x (Tensor): shape(batch_size, time_steps, feature_size), the input tensor.
            mask (Tensor): shape(batch_size, 1, time_steps), the mask.
            drop_n_heads (int, optional): how many heads to drop. Defaults to 0.

        Returns:
            x (Tensor): shape(batch_size, time_steps, feature_size), the context vector.
            attention_weights(list[Tensor]), each of shape
                (batch_size, n_heads, time_steps, time_steps), the attention weights.
        """
        attention_weights = []
        for layer in self:
            x, attention_weights_i = layer(x, mask, drop_n_heads)
            attention_weights.append(attention_weights_i)
        return x, attention_weights


class TransformerDecoder(nn.LayerList):
    def __init__(self,
                 d_model,
                 n_heads,
                 d_ffn,
                 n_layers,
                 dropout=0.,
                 d_encoder=None):
        super(TransformerDecoder, self).__init__()
        for _ in range(n_layers):
            self.append(
                TransformerDecoderLayer(
                    d_model, n_heads, d_ffn, dropout, d_encoder=d_encoder))

    def forward(self, q, k, v, encoder_mask, decoder_mask, drop_n_heads=0):
        """
        Args:
            q (Tensor): shape(batch_size, time_steps_q, d_model)
            k (Tensor): shape(batch_size, time_steps_k, d_encoder)
            v (Tensor): shape(batch_size, time_steps_k, k_encoder)
            encoder_mask (Tensor): shape(batch_size, 1, time_steps_k)
            decoder_mask (Tensor): shape(batch_size, time_steps_q, time_steps_q)
            drop_n_heads (int, optional): [description]. Defaults to 0.

        Returns:
            q (Tensor): shape(batch_size, time_steps_q, d_model), the output.
            self_attention_weights (List[Tensor]): shape (batch_size, num_heads, encoder_steps, encoder_steps)
            cross_attention_weights (List[Tensor]): shape (batch_size, num_heads, decoder_steps, encoder_steps)
        """
        self_attention_weights = []
        cross_attention_weights = []
        for layer in self:
            q, self_attention_weights_i, cross_attention_weights_i = layer(
                q, k, v, encoder_mask, decoder_mask, drop_n_heads)
            self_attention_weights.append(self_attention_weights_i)
            cross_attention_weights.append(cross_attention_weights_i)
        return q, self_attention_weights, cross_attention_weights


class MLPPreNet(nn.Layer):
    """Decoder's prenet."""

    def __init__(self, d_input, d_hidden, d_output, dropout):
        # (lin + relu + dropout) * n + last projection
        super(MLPPreNet, self).__init__()
        self.lin1 = nn.Linear(d_input, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_hidden)
        self.lin3 = nn.Linear(d_hidden, d_hidden)
        self.dropout = dropout

    def forward(self, x, dropout):
        l1 = F.dropout(
            F.relu(self.lin1(x)), self.dropout, training=self.training)
        l2 = F.dropout(
            F.relu(self.lin2(l1)), self.dropout, training=self.training)
        l3 = self.lin3(l2)
        return l3


class CNNPostNet(nn.Layer):
    def __init__(self, d_input, d_hidden, d_output, kernel_size, n_layers):
        super(CNNPostNet, self).__init__()
        self.convs = nn.LayerList()
        kernel_size = kernel_size if isinstance(kernel_size, (
            tuple, list)) else (kernel_size, )
        padding = (kernel_size[0] - 1, 0)
        for i in range(n_layers):
            c_in = d_input if i == 0 else d_hidden
            c_out = d_output if i == n_layers - 1 else d_hidden
            self.convs.append(
                Conv1dBatchNorm(
                    c_in,
                    c_out,
                    kernel_size,
                    weight_attr=I.XavierUniform(),
                    padding=padding))
        self.last_bn = nn.BatchNorm1D(d_output)
        # for a layer that ends with a normalization layer that is targeted to
        # output a non zero-central output, it may take a long time to 
        # train the scale and bias
        # NOTE: it can also be a non-causal conv

    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.convs):
            x = layer(x)
            if i != (len(self.convs) - 1):
                x = F.tanh(x)
        x = self.last_bn(x_in + x)
        return x


class TransformerTTS(nn.Layer):
    def __init__(self,
                 frontend: parakeet.frontend.Phonetics,
                 d_encoder: int,
                 d_decoder: int,
                 d_mel: int,
                 n_heads: int,
                 d_ffn: int,
                 encoder_layers: int,
                 decoder_layers: int,
                 d_prenet: int,
                 d_postnet: int,
                 postnet_layers: int,
                 postnet_kernel_size: int,
                 max_reduction_factor: int,
                 decoder_prenet_dropout: float,
                 dropout: float):
        super(TransformerTTS, self).__init__()

        # text frontend (text normalization and g2p)
        self.frontend = frontend

        # encoder
        self.encoder_prenet = nn.Embedding(
            frontend.vocab_size,
            d_encoder,
            padding_idx=frontend.vocab.padding_index,
            weight_attr=I.Uniform(-0.05, 0.05))
        # position encoding matrix may be extended later
        self.encoder_pe = pe.positional_encoding(0, 1000, d_encoder)
        self.encoder_pe_scalar = self.create_parameter(
            [1], attr=I.Constant(1.))
        self.encoder = TransformerEncoder(d_encoder, n_heads, d_ffn,
                                          encoder_layers, dropout)

        # decoder
        self.decoder_prenet = MLPPreNet(d_mel, d_prenet, d_decoder, dropout)
        self.decoder_pe = pe.positional_encoding(0, 1000, d_decoder)
        self.decoder_pe_scalar = self.create_parameter(
            [1], attr=I.Constant(1.))
        self.decoder = TransformerDecoder(
            d_decoder,
            n_heads,
            d_ffn,
            decoder_layers,
            dropout,
            d_encoder=d_encoder)
        self.final_proj = nn.Linear(d_decoder, max_reduction_factor * d_mel)
        self.decoder_postnet = CNNPostNet(d_mel, d_postnet, d_mel,
                                          postnet_kernel_size, postnet_layers)
        self.stop_conditioner = nn.Linear(d_mel, 3)

        # specs
        self.padding_idx = frontend.vocab.padding_index
        self.d_encoder = d_encoder
        self.d_decoder = d_decoder
        self.d_mel = d_mel
        self.max_r = max_reduction_factor
        self.dropout = dropout
        self.decoder_prenet_dropout = decoder_prenet_dropout

        # start and end: though it is only used in predict 
        # it can also be used in training
        dtype = paddle.get_default_dtype()
        self.start_vec = paddle.full([1, d_mel], 0.5, dtype=dtype)
        self.end_vec = paddle.full([1, d_mel], -0.5, dtype=dtype)
        self.stop_prob_index = 2

        # mutables
        self.r = max_reduction_factor  # set it every call
        self.drop_n_heads = 0

    def forward(self, text, mel):
        encoded, encoder_attention_weights, encoder_mask = self.encode(text)
        mel_output, mel_intermediate, cross_attention_weights, stop_logits = self.decode(
            encoded, mel, encoder_mask)
        outputs = {
            "mel_output": mel_output,
            "mel_intermediate": mel_intermediate,
            "encoder_attention_weights": encoder_attention_weights,
            "cross_attention_weights": cross_attention_weights,
            "stop_logits": stop_logits,
        }
        return outputs

    def encode(self, text):
        T_enc = text.shape[-1]
        embed = self.encoder_prenet(text)
        if embed.shape[1] > self.encoder_pe.shape[0]:
            new_T = max(embed.shape[1], self.encoder_pe.shape[0] * 2)
            self.encoder_pe = pe.positional_encoding(0, new_T, self.d_encoder)
        pos_enc = self.encoder_pe[:T_enc, :]  # (T, C)
        x = embed.scale(math.sqrt(
            self.d_encoder)) + pos_enc * self.encoder_pe_scalar
        x = F.dropout(x, self.dropout, training=self.training)

        # TODO(chenfeiyu): unsqueeze a decoder_time_steps=1 for the mask
        encoder_padding_mask = paddle.unsqueeze(
            masking.id_mask(
                text, self.padding_idx, dtype=x.dtype), 1)
        x, attention_weights = self.encoder(x, encoder_padding_mask,
                                            self.drop_n_heads)
        return x, attention_weights, encoder_padding_mask

    def decode(self, encoder_output, input, encoder_padding_mask):
        batch_size, T_dec, mel_dim = input.shape

        x = self.decoder_prenet(input, self.decoder_prenet_dropout)
        # twice its length if needed
        if x.shape[1] * self.r > self.decoder_pe.shape[0]:
            new_T = max(x.shape[1] * self.r, self.decoder_pe.shape[0] * 2)
            self.decoder_pe = pe.positional_encoding(0, new_T, self.d_decoder)
        pos_enc = self.decoder_pe[:T_dec * self.r:self.r, :]
        x = x.scale(math.sqrt(
            self.d_decoder)) + pos_enc * self.decoder_pe_scalar
        x = F.dropout(x, self.dropout, training=self.training)

        no_future_mask = masking.future_mask(T_dec, dtype=input.dtype)
        decoder_padding_mask = masking.feature_mask(
            input, axis=-1, dtype=input.dtype)
        decoder_mask = masking.combine_mask(
            decoder_padding_mask.unsqueeze(-1), no_future_mask)
        decoder_output, _, cross_attention_weights = self.decoder(
            x, encoder_output, encoder_output, encoder_padding_mask,
            decoder_mask, self.drop_n_heads)

        # use only parts of it
        output_proj = self.final_proj(decoder_output)[:, :, :self.r * mel_dim]
        mel_intermediate = paddle.reshape(output_proj,
                                          [batch_size, -1, mel_dim])
        stop_logits = self.stop_conditioner(mel_intermediate)

        # cnn postnet
        mel_channel_first = paddle.transpose(mel_intermediate, [0, 2, 1])
        mel_output = self.decoder_postnet(mel_channel_first)
        mel_output = paddle.transpose(mel_output, [0, 2, 1])

        return mel_output, mel_intermediate, cross_attention_weights, stop_logits

    @paddle.no_grad()
    def infer(self, input, max_length=1000, verbose=True):
        """Predict log scale magnitude mel spectrogram from text input.

        Args:
            input (Tensor): shape (T), dtype int, input text sequencce.
            max_length (int, optional): max decoder steps. Defaults to 1000.
            verbose (bool, optional): display progress bar. Defaults to True.
        """
        decoder_input = paddle.unsqueeze(self.start_vec, 0)  # (B=1, T, C)
        decoder_output = paddle.unsqueeze(self.start_vec, 0)  # (B=1, T, C)

        # encoder the text sequence
        encoder_output, encoder_attentions, encoder_padding_mask = self.encode(
            input)
        for _ in trange(int(max_length // self.r) + 1):
            mel_output, _, cross_attention_weights, stop_logits = self.decode(
                encoder_output, decoder_input, encoder_padding_mask)

            # extract last step and append it to decoder input
            decoder_input = paddle.concat(
                [decoder_input, mel_output[:, -1:, :]], 1)
            # extract last r steps and append it to decoder output
            decoder_output = paddle.concat(
                [decoder_output, mel_output[:, -self.r:, :]], 1)

            # stop condition: (if any ouput frame of the output multiframes hits the stop condition)
            if paddle.any(
                    paddle.argmax(
                        stop_logits[0, -self.r:, :], axis=-1) ==
                    self.stop_prob_index):
                if verbose:
                    print("Hits stop condition.")
                break
        mel_output = decoder_output[:, 1:, :]

        outputs = {
            "mel_output": mel_output,
            "encoder_attention_weights": encoder_attentions,
            "cross_attention_weights": cross_attention_weights,
        }
        return outputs

    @paddle.no_grad()
    def predict(self, input, max_length=1000, verbose=True):
        text_ids = paddle.to_tensor(self.frontend(input))
        input = paddle.unsqueeze(text_ids, 0)  # (1, T)
        outputs = self.infer(input, max_length=max_length, verbose=verbose)
        outputs = {k: v[0].numpy() for k, v in outputs.items()}
        return outputs

    def set_constants(self, reduction_factor, drop_n_heads):
        self.r = reduction_factor
        self.drop_n_heads = drop_n_heads

    @classmethod
    def from_pretrained(cls, frontend, config, checkpoint_path):
        model = TransformerTTS(
            frontend,
            d_encoder=config.model.d_encoder,
            d_decoder=config.model.d_decoder,
            d_mel=config.data.d_mel,
            n_heads=config.model.n_heads,
            d_ffn=config.model.d_ffn,
            encoder_layers=config.model.encoder_layers,
            decoder_layers=config.model.decoder_layers,
            d_prenet=config.model.d_prenet,
            d_postnet=config.model.d_postnet,
            postnet_layers=config.model.postnet_layers,
            postnet_kernel_size=config.model.postnet_kernel_size,
            max_reduction_factor=config.model.max_reduction_factor,
            decoder_prenet_dropout=config.model.decoder_prenet_dropout,
            dropout=config.model.dropout)

        iteration = checkpoint.load_parameters(
            model, checkpoint_path=checkpoint_path)
        drop_n_heads = scheduler.StepWise(config.training.drop_n_heads)
        reduction_factor = scheduler.StepWise(config.training.reduction_factor)
        model.set_constants(
            reduction_factor=reduction_factor(iteration),
            drop_n_heads=drop_n_heads(iteration))
        return model


class TransformerTTSLoss(nn.Layer):
    def __init__(self, stop_loss_scale):
        super(TransformerTTSLoss, self).__init__()
        self.stop_loss_scale = stop_loss_scale

    def forward(self, mel_output, mel_intermediate, mel_target, stop_logits,
                stop_probs):
        mask = masking.feature_mask(
            mel_target, axis=-1, dtype=mel_target.dtype)
        mask1 = paddle.unsqueeze(mask, -1)
        mel_loss1 = L.masked_l1_loss(mel_output, mel_target, mask1)
        mel_loss2 = L.masked_l1_loss(mel_intermediate, mel_target, mask1)

        mel_len = mask.shape[-1]
        last_position = F.one_hot(
            mask.sum(-1).astype("int64") - 1, num_classes=mel_len)
        mask2 = mask + last_position.scale(self.stop_loss_scale - 1).astype(
            mask.dtype)
        stop_loss = L.masked_softmax_with_cross_entropy(
            stop_logits, stop_probs.unsqueeze(-1), mask2.unsqueeze(-1))

        loss = mel_loss1 + mel_loss2 + stop_loss
        losses = dict(
            loss=loss,  # total loss
            mel_loss1=mel_loss1,  # ouput mel loss
            mel_loss2=mel_loss2,  # intermediate mel loss
            stop_loss=stop_loss  # stop prob loss
        )
        return losses
