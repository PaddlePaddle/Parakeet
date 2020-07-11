import numpy as np
import math
import paddle
from paddle import fluid
from paddle.fluid import layers as F
from paddle.fluid import initializer as I
from paddle.fluid import dygraph as dg

from .conv import Conv1D
from .weight_norm_hook import weight_norm, remove_weight_norm

def positional_encoding(tensor, start_index, omega):
    """
    tensor: a reference tensor we use to get shape. actually only T and C are needed. Shape(B, T, C)
    start_index: int, we can actually use start and length to specify them.
    omega (B,): speaker position rates

    return (B, T, C), position embedding
    """
    dtype = omega.dtype
    _, length, dimension = tensor.shape
    index = F.range(start_index, start_index + length, 1, dtype=dtype)
    channel = F.range(0, dimension, 2, dtype=dtype)

    p = F.unsqueeze(omega, [1, 2]) \
      * F.unsqueeze(index, [1]) \
      / (10000 ** (channel / float(dimension)))

    encodings = F.concat([F.sin(p), F.cos(p)], axis=2)
    return encodings

class ConvBlock(dg.Layer):
    def __init__(self, in_channel, kernel_size, causal=False, has_bias=False, 
                 bias_dim=None, keep_prob=1.):
        super(ConvBlock, self).__init__()
        self.causal = causal
        self.keep_prob = keep_prob
        self.in_channel = in_channel
        self.has_bias = has_bias

        std = np.sqrt(4 * keep_prob / (kernel_size * in_channel))
        initializer = I.NormalInitializer(loc=0., scale=std)
        padding = "valid" if causal else "same"
        conv =  Conv1D(in_channel, 2 * in_channel, (kernel_size, ),
                       padding=padding, 
                       data_format="NTC",
                       param_attr=initializer)
        self.conv = weight_norm(conv)
        if has_bias:
            self.bias_affine = dg.Linear(bias_dim, 2 * in_channel)

    def forward(self, input, bias=None, padding=None):
        """
        input: input feature (B, T, C)
        padding: only used when using causal conv, we pad mannually
        """
        input_dropped = F.dropout(input, 1. - self.keep_prob,
                                  dropout_implementation="upscale_in_train")
        if self.causal:
            assert padding is not None
            input_dropped = F.concat([padding, input_dropped], axis=1)
        hidden = self.conv(input_dropped)

        if self.has_bias:
            assert bias is not None
            transformed_bias = F.softsign(self.bias_affine(bias))
            hidden_embedded = hidden + F.unsqueeze(transformed_bias, [1])
        else:
            hidden_embedded = hidden

        # glu
        content, gate = F.split(hidden, num_or_sections=2, dim=-1)
        content = hidden_embedded[:, :, :self.in_channel]
        hidden = F.sigmoid(gate) * content

        # # residual
        hidden = F.scale(input + hidden, math.sqrt(0.5))
        return hidden


class AffineBlock1(dg.Layer):
    def __init__(self, in_channel, out_channel, has_bias=False, bias_dim=0):
        super(AffineBlock1, self).__init__()
        std = np.sqrt(1.0 / in_channel)
        initializer = I.NormalInitializer(loc=0., scale=std)
        affine = dg.Linear(in_channel, out_channel, param_attr=initializer)
        self.affine = weight_norm(affine, dim=-1)
        if has_bias:
            self.bias_affine = dg.Linear(bias_dim, out_channel)

        self.has_bias = has_bias
        self.bias_dim = bias_dim

    def forward(self, input, bias=None):
        """
        input -> (affine + weight_norm) ->hidden
        bias -> (affine) -> softsign -> transformed_bis
        hidden += transformed_bias
        """
        hidden = self.affine(input)
        if self.has_bias:
            assert bias is not None
            transformed_bias = F.softsign(self.bias_affine(bias))
            hidden += F.unsqueeze(transformed_bias, [1])
        return hidden


class AffineBlock2(dg.Layer):
    def __init__(self, in_channel, out_channel,
                 has_bias=False, bias_dim=0, dropout=False, keep_prob=1.):
        super(AffineBlock2, self).__init__()
        if has_bias:
            self.bias_affine = dg.Linear(bias_dim, in_channel)
        std = np.sqrt(1.0 / in_channel)
        initializer = I.NormalInitializer(loc=0., scale=std)
        affine = dg.Linear(in_channel, out_channel, param_attr=initializer)
        self.affine = weight_norm(affine, dim=-1)

        self.has_bias = has_bias
        self.bias_dim = bias_dim
        self.dropout = dropout
        self.keep_prob = keep_prob

    def forward(self, input, bias=None):
        """
        input -> (dropout) ->hidden
        bias -> (affine) -> softsign -> transformed_bis
        hidden += transformed_bias
        hidden -> (affine + weight_norm) -> relu -> hidden
        """
        hidden = input
        if self.dropout:
            hidden = F.dropout(hidden, 1. - self.keep_prob,
                               dropout_implementation="upscale_in_train")
        if self.has_bias:
            assert bias is not None
            transformed_bias = F.softsign(self.bias_affine(bias))
            hidden += F.unsqueeze(transformed_bias, [1])
        hidden = F.relu(self.affine(hidden))
        return hidden


class Encoder(dg.Layer):
    def __init__(self, layers, in_channels, encoder_dim, kernel_size, 
                 has_bias=False, bias_dim=0, keep_prob=1.):
        super(Encoder, self).__init__()
        self.pre_affine = AffineBlock1(in_channels, encoder_dim, has_bias, bias_dim)
        self.convs = dg.LayerList([
            ConvBlock(encoder_dim, kernel_size, False, has_bias, bias_dim, keep_prob) \
                for _ in range(layers)])
        self.post_affine = AffineBlock1(encoder_dim, in_channels, has_bias, bias_dim)
        
    def forward(self, char_embed, speaker_embed=None):
        hidden = self.pre_affine(char_embed, speaker_embed)
        for layer in self.convs:
            hidden = layer(hidden, speaker_embed)
        hidden = self.post_affine(hidden, speaker_embed)
        keys = hidden
        values = F.scale(char_embed + hidden, np.sqrt(0.5))
        return keys, values


class AttentionBlock(dg.Layer):
    def __init__(self, attention_dim, input_dim, position_encoding_weight=1., 
                 position_rate=1., reduction_factor=1, has_bias=False, bias_dim=0, 
                 keep_prob=1.):
        super(AttentionBlock, self).__init__()
        # positional encoding
        omega_default = position_rate / reduction_factor
        self.omega_default = omega_default
        # multispeaker case
        if has_bias:
            std = np.sqrt(1.0 / bias_dim)
            initializer = I.NormalInitializer(loc=0., scale=std)
            self.q_pos_affine = dg.Linear(bias_dim, 1, param_attr=initializer)
            self.k_pos_affine = dg.Linear(bias_dim, 1, param_attr=initializer)
            self.omega_initial = self.create_parameter(shape=[1], 
                attr=I.ConstantInitializer(value=omega_default))
        
        # mind the fact that q, k, v have the same feature dimension
        # so we can init k_affine and q_affine's weight as the same matrix
        # to get a better init attention
        init_weight = np.random.normal(size=(input_dim, attention_dim),
                                       scale=np.sqrt(1. / input_dim))
        initializer = I.NumpyArrayInitializer(init_weight.astype(np.float32))
        # 3 affine transformation to project q, k, v into attention_dim
        q_affine = dg.Linear(input_dim, attention_dim, 
                                    param_attr=initializer)
        self.q_affine = weight_norm(q_affine, dim=-1)
        k_affine = dg.Linear(input_dim, attention_dim, 
                                    param_attr=initializer)
        self.k_affine = weight_norm(k_affine, dim=-1)

        std = np.sqrt(1.0 / input_dim)
        initializer = I.NormalInitializer(loc=0., scale=std)
        v_affine = dg.Linear(input_dim, attention_dim, param_attr=initializer)
        self.v_affine = weight_norm(v_affine, dim=-1)

        std = np.sqrt(1.0 / attention_dim)
        initializer = I.NormalInitializer(loc=0., scale=std)
        out_affine = dg.Linear(attention_dim, input_dim, param_attr=initializer)
        self.out_affine = weight_norm(out_affine, dim=-1)

        self.keep_prob = keep_prob
        self.has_bias = has_bias
        self.bias_dim = bias_dim
        self.attention_dim = attention_dim
        self.position_encoding_weight = position_encoding_weight

    def forward(self, q, k, v, lengths, speaker_embed, start_index, 
                force_monotonic=False, prev_coeffs=None, window=None):
        # add position encoding as an inductive bias 
        if self.has_bias: # multi-speaker model
            omega_q = 2 * F.sigmoid(
                F.squeeze(self.q_pos_affine(speaker_embed), axes=[-1]))
            omega_k = 2 * self.omega_initial * F.sigmoid(F.squeeze(
                self.k_pos_affine(speaker_embed), axes=[-1]))
        else: # single-speaker case
            batch_size = q.shape[0]
            omega_q = F.ones((batch_size, ), dtype="float32")
            omega_k = F.ones((batch_size, ), dtype="float32") * self.omega_default
        q += self.position_encoding_weight * positional_encoding(q, start_index, omega_q)
        k += self.position_encoding_weight * positional_encoding(k, 0, omega_k)

        q, k, v = self.q_affine(q), self.k_affine(k), self.v_affine(v)
        activations = F.matmul(q, k, transpose_y=True)
        activations /= np.sqrt(self.attention_dim)

        if self.training:
            # mask the <pad> parts from the encoder
            mask = F.sequence_mask(lengths, dtype="float32")
            attn_bias = F.scale(1. - mask, -1000)
            activations += F.unsqueeze(attn_bias, [1])
        elif force_monotonic:
            assert window is not None
            backward_step, forward_step = window
            T_enc = k.shape[1]
            batch_size, T_dec, _ = q.shape

            # actually T_dec = 1 here
            alpha = F.fill_constant((batch_size, T_dec), value=0, dtype="int64") \
                   if prev_coeffs is None \
                   else F.argmax(prev_coeffs, axis=-1)
            backward = F.sequence_mask(alpha - backward_step, maxlen=T_enc, dtype="bool")
            forward = F.sequence_mask(alpha + forward_step, maxlen=T_enc, dtype="bool")
            mask = F.cast(F.logical_xor(backward, forward), "float32")
            # print("mask's shape:", mask.shape)
            attn_bias = F.scale(1. - mask, -1000)
            activations += attn_bias

        # softmax
        coefficients = F.softmax(activations, axis=-1)
        # context vector
        coefficients = F.dropout(coefficients, 1. - self.keep_prob,
                                 dropout_implementation='upscale_in_train')
        contexts = F.matmul(coefficients, v)
        # context normalization
        enc_lengths = F.cast(F.unsqueeze(lengths, axes=[1, 2]), "float32")
        contexts *= F.sqrt(enc_lengths)
        # out affine
        contexts = self.out_affine(contexts)
        return contexts, coefficients


class Decoder(dg.Layer):
    def __init__(self, in_channels, reduction_factor, prenet_sizes, 
                layers, kernel_size, attention_dim,
                position_encoding_weight=1., omega=1., 
                has_bias=False, bias_dim=0, keep_prob=1.):
        super(Decoder, self).__init__()
        # prenet-mind the difference of AffineBlock2 and AffineBlock1
        c_in = in_channels
        self.prenet = dg.LayerList()
        for i, c_out in enumerate(prenet_sizes):
            affine = AffineBlock2(c_in, c_out, has_bias, bias_dim, dropout=(i!=0), keep_prob=keep_prob)
            self.prenet.append(affine)
            c_in = c_out
        
        # causal convolutions + multihop attention
        decoder_dim = prenet_sizes[-1]
        self.causal_convs = dg.LayerList()
        self.attention_blocks = dg.LayerList()
        for i in range(layers):
            conv = ConvBlock(decoder_dim, kernel_size, True, has_bias, bias_dim, keep_prob)
            attn = AttentionBlock(attention_dim, decoder_dim, position_encoding_weight, omega, reduction_factor, has_bias, bias_dim, keep_prob)
            self.causal_convs.append(conv)
            self.attention_blocks.append(attn)

        # output mel spectrogram
        output_dim = reduction_factor * in_channels # r * mel_dim
        std = np.sqrt(1.0 / decoder_dim)
        initializer = I.NormalInitializer(loc=0., scale=std)
        out_affine = dg.Linear(decoder_dim, output_dim, param_attr=initializer)
        self.out_affine = weight_norm(out_affine, dim=-1)
        if has_bias:
            self.out_sp_affine = dg.Linear(bias_dim, output_dim)

        self.has_bias = has_bias
        self.kernel_size = kernel_size

        self.in_channels = in_channels
        self.decoder_dim = decoder_dim
        self.reduction_factor = reduction_factor
        self.out_channels = output_dim

    def forward(self, inputs, keys, values, lengths, start_index, speaker_embed=None, 
                state=None, force_monotonic_attention=None, coeffs=None, window=(0, 4)):
        hidden = inputs
        for layer in self.prenet:
            hidden = layer(hidden, speaker_embed)

        attentions = [] # every layer of (B, T_dec, T_enc) attention
        final_state = [] # layers * (B, (k-1)d, C_dec)
        batch_size = inputs.shape[0]
        causal_padding_shape = (batch_size, self.kernel_size - 1, self.decoder_dim)

        for i in range(len(self.causal_convs)):
            if state is None:
                padding = F.zeros(causal_padding_shape, dtype="float32")
            else:
                padding = state[i]
            new_state = F.concat([padding, hidden], axis=1) # => to be used next step
            # causal conv, (B, T, C)
            hidden = self.causal_convs[i](hidden, speaker_embed, padding=padding)
            # attn
            prev_coeffs = None if coeffs is None else coeffs[i] 
            force_monotonic = False if force_monotonic_attention is None else force_monotonic_attention[i]
            context, attention = self.attention_blocks[i](
                hidden, keys, values, lengths, speaker_embed, 
                start_index, force_monotonic, prev_coeffs, window)
            # residual connextion (B, T_dec, C_dec)
            hidden = F.scale(hidden + context, np.sqrt(0.5))

            attentions.append(attention) # layers * (B, T_dec, T_enc)
            # new state: shift a step, layers * (B, T, C)
            new_state = new_state[:, -(self.kernel_size - 1):, :]
            final_state.append(new_state)

        # predict mel spectrogram (B, 1, T_dec, r * C_in)
        decoded = self.out_affine(hidden)
        if self.has_bias:
            decoded *= F.sigmoid(F.unsqueeze(self.out_sp_affine(speaker_embed), [1]))
        return decoded, hidden, attentions, final_state


class PostNet(dg.Layer):
    def __init__(self, layers, in_channels, postnet_dim, kernel_size, out_channels, upsample_factor, has_bias=False, bias_dim=0, keep_prob=1.):
        super(PostNet, self).__init__()
        self.pre_affine = AffineBlock1(in_channels, postnet_dim, has_bias, bias_dim)
        self.convs = dg.LayerList([
            ConvBlock(postnet_dim, kernel_size, False, has_bias, bias_dim, keep_prob) for _ in range(layers)
        ])
        std = np.sqrt(1.0 / postnet_dim)
        initializer = I.NormalInitializer(loc=0., scale=std)
        post_affine = dg.Linear(postnet_dim, out_channels, param_attr=initializer)
        self.post_affine = weight_norm(post_affine, dim=-1)
        self.upsample_factor = upsample_factor

    def forward(self, hidden, speaker_embed=None):
        hidden = self.pre_affine(hidden, speaker_embed)
        batch_size, time_steps, channels = hidden.shape # pylint: disable=unused-variable
        hidden = F.expand(hidden, [1, 1, self.upsample_factor])
        hidden = F.reshape(hidden, [batch_size, -1, channels])
        for layer in self.convs:
            hidden = layer(hidden, speaker_embed)
        spec = self.post_affine(hidden)
        return spec


class SpectraNet(dg.Layer):
    def __init__(self, char_embedding, speaker_embedding, encoder, decoder, postnet):
        super(SpectraNet, self).__init__()
        self.char_embedding = char_embedding
        self.speaker_embedding = speaker_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.postnet = postnet
    
    def forward(self, text, text_lengths, speakers=None, mel=None, frame_lengths=None, 
                force_monotonic_attention=None, window=None):
        # encode
        text_embed = self.char_embedding(text)# no stress embedding here
        speaker_embed = F.softsign(self.speaker_embedding(speakers)) if self.speaker_embedding is not None else None
        keys, values = self.encoder(text_embed, speaker_embed)

        if mel is not None:
            return self.teacher_forced_train(keys, values, text_lengths, speaker_embed, mel)
        else:
            return self.inference(keys, values, text_lengths, speaker_embed, force_monotonic_attention, window)

    def teacher_forced_train(self, keys, values, text_lengths, speaker_embed, mel):
        # build decoder inputs by shifting over by one frame and add all zero <start> frame
        # the mel input is downsampled by a reduction factor
        batch_size = mel.shape[0]
        mel_input = F.reshape(mel, (batch_size, -1, self.decoder.reduction_factor, self.decoder.in_channels))
        zero_frame = F.zeros((batch_size, 1, self.decoder.in_channels), dtype="float32")
        # downsample mel input as a regularization
        mel_input = F.concat([zero_frame, mel_input[:, :-1, -1, :]], axis=1)

        # decoder
        decoded, hidden, attentions, final_state = self.decoder(mel_input, keys, values, text_lengths, 0, speaker_embed)
        attentions = F.stack(attentions) # (N, B, T_dec, T_encs)
        # unfold frames
        decoded = F.reshape(decoded, (batch_size, -1, self.decoder.in_channels))
        # postnet
        refined = self.postnet(hidden, speaker_embed)
        return decoded, refined, attentions, final_state

    def spec_loss(self, decoded, input, num_frames=None):
        if num_frames is None:
            l1_loss = F.reduce_mean(F.abs(decoded - input))
        else:
            # mask the <pad> part of the decoder
            num_channels = decoded.shape[-1]
            l1_loss = F.abs(decoded - input)
            mask = F.sequence_mask(num_frames, dtype="float32")
            l1_loss *= F.unsqueeze(mask, axes=[-1])
            l1_loss = F.reduce_sum(l1_loss) / F.scale(F.reduce_sum(mask), num_channels)
        return l1_loss

    @dg.no_grad
    def inference(self, keys, values, text_lengths, speaker_embed, 
                  force_monotonic_attention, window):
        MAX_STEP = 500
        
        # layer index of the first monotonic attention
        num_monotonic_attention_layers = sum(force_monotonic_attention)
        first_mono_attention_layer = 0
        if num_monotonic_attention_layers > 0:
            for i, item in enumerate(force_monotonic_attention):
                if item:
                    first_mono_attention_layer = i
                    break
            
        # stop cond (if would be more complicated to support minibatch autoregressive decoding)
        # so we only supports batch_size == 0 in inference
        def should_continue(i, mel_input, outputs, hidden, attention, state, coeffs):
            T_enc = coeffs.shape[-1]
            attn_peak = F.argmax(coeffs[first_mono_attention_layer, 0, 0]) \
                if num_monotonic_attention_layers > 0 \
                else F.fill_constant([1], "int64", value=0)
            return i < MAX_STEP and F.reshape(attn_peak, [1]) < T_enc - 1
        
        def loop_body(i, mel_input, outputs, hiddens, attentions, state=None, coeffs=None):
            # state is None coeffs is None for the first step
            decoded, hidden, new_coeffs, new_state = self.decoder(
                mel_input, keys, values, text_lengths, i, speaker_embed, 
                state, force_monotonic_attention, coeffs, window)
            new_coeffs = F.stack(new_coeffs) # (N, B, T_dec=1, T_enc)

            attentions.append(new_coeffs) # (N, B, T_dec=1, T_enc)
            outputs.append(decoded) # (B, T_dec=1, rC_mel)
            hiddens.append(hidden) # (B, T_dec=1, C_dec)

            # slice the last frame out of r generated frames to be used as the input for the next step
            batch_size = mel_input.shape[0]
            frames = F.reshape(decoded, [batch_size, -1, self.decoder.reduction_factor, self.decoder.in_channels])
            input_frame = frames[:, :, -1, :]
            return (i + 1, input_frame, outputs, hiddens, attentions, new_state, new_coeffs)

        i = 0
        batch_size = keys.shape[0]
        input_frame = F.zeros((batch_size, 1, self.decoder.in_channels), dtype="float32")
        outputs = []
        hiddens = []
        attentions = []
        loop_state = loop_body(i, input_frame, outputs, hiddens, attentions)

        while should_continue(*loop_state):
            loop_state = loop_body(*loop_state)
    
        outputs, hiddens, attention = loop_state[2], loop_state[3], loop_state[4]
        # concat decoder timesteps
        outputs = F.concat(outputs, axis=1)
        hiddens = F.concat(hiddens, axis=1)
        attention = F.concat(attention, axis=2)

        # unfold frames
        outputs = F.reshape(outputs, (batch_size, -1, self.decoder.in_channels))

        refined = self.postnet(hiddens, speaker_embed)
        return outputs, refined, attention
