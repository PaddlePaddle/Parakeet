import numpy as np
import paddle.fluid.layers as F
import paddle.fluid.initializer as I
import paddle.fluid.dygraph as dg

from parakeet.modules.weight_norm import Conv1D, Linear
from parakeet.models.deepvoice3.conv1dglu import Conv1DGLU
from parakeet.models.deepvoice3.encoder import ConvSpec
from parakeet.models.deepvoice3.attention import Attention, WindowRange
from parakeet.models.deepvoice3.position_embedding import PositionEmbedding


def gen_mask(valid_lengths, max_len, dtype="float32"):
    """
    Generate a mask tensor from valid lengths. note that it return a *reverse*
    mask. Indices within valid lengths correspond to 0, and those within
    padding area correspond to 1. 
    
    Assume that valid_lengths = [2,5,7], and max_len = 7, the generated mask is
    [[0, 0, 1, 1, 1, 1, 1],
     [0, 0, 0, 0, 0, 1, 1],
     [0, 0, 0, 0, 0, 0, 0]].

    Args:
        valid_lengths (Variable): Shape(B), dtype: int64. A 1D-Tensor containing
            the valid lengths (timesteps) of each example, where B means
            beatch_size.
        max_len (int): The length (number of timesteps) of the mask.
        dtype (str, optional): A string that specifies the data type of the
            returned mask.

    Returns:
        mask (Variable): A mask computed from valid lengths.
    """
    mask = F.sequence_mask(valid_lengths, maxlen=max_len, dtype=dtype)
    mask = 1 - mask
    return mask


def fold_adjacent_frames(frames, r):
    """fold multiple adjacent frames.
    
    Arguments:
        frames {Variable} -- shape(batch_size, time_steps, channels), the spectrogram
        r {int} -- frames per step.
    
    Returns:
        Variable -- shape(batch_size, time_steps // r, r *channels), folded frames
    """

    if r == 1:
        return frames
    batch_size, time_steps, channels = frames.shape
    if time_steps % r != 0:
        print(
            "time_steps cannot be divided by r, you would lose {} tailing frames"
            .format(time_steps % r))
        frames = frames[:, :time_steps - time_steps % r, :]
    frames = F.reshape(frames, (batch_size, -1, channels * r))
    return frames


def unfold_adjacent_frames(folded_frames, r):
    """fold multiple adjacent frames.
    
    Arguments:
        folded_frames {Variable} -- shape(batch_size, time_steps // r, r * channels), the spectrogram
        r {int} -- frames per step.
    
    Returns:
        Variable -- shape(batch_size, time_steps, channels), folded frames
    """

    if r == 1:
        return folded_frames
    batch_size, time_steps, channels = folded_frames.shape
    folded_frames = F.reshape(folded_frames, (batch_size, -1, channels // r))
    return folded_frames


class Decoder(dg.Layer):
    def __init__(
        self,
        n_speakers,
        speaker_dim,
        embed_dim,
        mel_dim,
        r=1,
        max_positions=512,
        padding_idx=None,  # remove it!
        preattention=(ConvSpec(128, 5, 1), ) * 4,
        convolutions=(ConvSpec(128, 5, 1), ) * 4,
        attention=True,
        dropout=0.0,
        use_memory_mask=False,
        force_monotonic_attention=False,
        query_position_rate=1.0,
        key_position_rate=1.0,
        window_range=WindowRange(-1, 3),
        key_projection=True,
        value_projection=True):
        super(Decoder, self).__init__()

        self.dropout = dropout
        self.mel_dim = mel_dim
        self.r = r
        self.query_position_rate = query_position_rate
        self.key_position_rate = key_position_rate
        self.window_range = window_range
        self.n_speakers = n_speakers

        conv_channels = convolutions[0].out_channels
        # only when padding idx is 0 can we easilt handle it
        self.embed_keys_positions = PositionEmbedding(max_positions,
                                                      embed_dim,
                                                      padding_idx=0)
        self.embed_query_positions = PositionEmbedding(max_positions,
                                                       conv_channels,
                                                       padding_idx=0)

        if n_speakers > 1:
            std = np.sqrt((1 - dropout) / speaker_dim)
            self.speaker_proj1 = Linear(speaker_dim,
                                        1,
                                        act="sigmoid",
                                        param_attr=I.Normal(scale=std))
            self.speaker_proj2 = Linear(speaker_dim,
                                        1,
                                        act="sigmoid",
                                        param_attr=I.Normal(scale=std))

        # prenet
        self.prenet = dg.LayerList()
        in_channels = mel_dim * r  # multiframe
        std_mul = 1.0
        for (out_channels, filter_size, dilation) in preattention:
            if in_channels != out_channels:
                # conv1d & relu
                std = np.sqrt(std_mul / in_channels)
                self.prenet.append(
                    Conv1D(in_channels,
                           out_channels,
                           1,
                           act="relu",
                           param_attr=I.Normal(scale=std)))
                in_channels = out_channels
                std_mul = 2.0
            self.prenet.append(
                Conv1DGLU(n_speakers,
                          speaker_dim,
                          in_channels,
                          out_channels,
                          filter_size,
                          dilation,
                          std_mul,
                          dropout,
                          causal=True,
                          residual=True))
            in_channels = out_channels
            std_mul = 4.0

        # attention
        self.use_memory_mask = use_memory_mask
        if isinstance(attention, bool):
            self.attention = [attention] * len(convolutions)
        else:
            self.attention = attention

        if isinstance(force_monotonic_attention, bool):
            self.force_monotonic_attention = [force_monotonic_attention
                                              ] * len(convolutions)
        else:
            self.force_monotonic_attention = force_monotonic_attention

        for x, y in zip(self.force_monotonic_attention, self.attention):
            if x is True and y is False:
                raise ValueError("When not using attention, there is no "
                                 "monotonic attention at all")

        # causual convolution & attention
        self.conv_attn = []
        for use_attention, (out_channels, filter_size,
                            dilation) in zip(self.attention, convolutions):
            assert (
                in_channels == out_channels
            ), "the stack of convolution & attention does not change channels"
            conv_layer = Conv1DGLU(n_speakers,
                                   speaker_dim,
                                   in_channels,
                                   out_channels,
                                   filter_size,
                                   dilation,
                                   std_mul,
                                   dropout,
                                   causal=True,
                                   residual=False)
            attn_layer = Attention(
                out_channels,
                embed_dim,
                dropout,
                window_range,
                key_projection=key_projection,
                value_projection=value_projection) if use_attention else None
            in_channels = out_channels
            std_mul = 4.0
            self.conv_attn.append((conv_layer, attn_layer))
        for i, (conv_layer, attn_layer) in enumerate(self.conv_attn):
            self.add_sublayer("conv_{}".format(i), conv_layer)
            if attn_layer is not None:
                self.add_sublayer("attn_{}".format(i), attn_layer)

        # 1 * 1 conv to transform channels
        std = np.sqrt(std_mul * (1 - dropout) / in_channels)
        self.last_conv = Conv1D(in_channels,
                                mel_dim * r,
                                1,
                                param_attr=I.Normal(scale=std))

        # mel (before sigmoid) to done hat
        std = np.sqrt(1 / in_channels)
        self.fc = Conv1D(mel_dim * r, 1, 1, param_attr=I.Normal(scale=std))

        # decoding configs
        self.max_decoder_steps = 200
        self.min_decoder_steps = 10

        assert convolutions[-1].out_channels % r == 0, \
                "decoder_state dim must be divided by r"
        self.state_dim = convolutions[-1].out_channels // self.r

    def forward(self,
                encoder_out,
                lengths,
                frames,
                text_positions,
                frame_positions,
                speaker_embed=None):
        """
        Compute decoder outputs with ground truth mel spectrogram.

        Args:
            encoder_out (Tuple(Variable, Variable)): 
                keys (Variable): shape(B, T_enc, C_emb), the key
                    representation from an encoder, where C_emb means
                    text embedding size.
                values (Variable): shape(B, T_enc, C_emb), the value
                    representation from an encoder, where C_emb means
                    text embedding size.
            lengths (Variable): shape(batch_size,), dtype: int64, valid lengths
                of text inputs for each example.
            inputs (Variable): shape(B, T_mel, C_mel), ground truth
                mel-spectrogram, which is used as decoder inputs when training.
            text_positions (Variable): shape(B, T_enc), dtype: int64.
                Positions indices for text inputs for the encoder, where 
                T_enc means the encoder timesteps.
            frame_positions (Variable): shape(B, T_mel // r), dtype: 
                int64. Positions indices for each decoder time steps.
            speaker_embed: shape(batch_size, speaker_dim), speaker embedding, 
                only used for multispeaker model.


        Returns:
            outputs (Variable): Shape(B, T_mel // r, r * C_mel). Decoder
                outputs, where C_mel means the channels of mel-spectrogram, r 
                means the outputs per decoder step, T_mel means the length(time
                steps) of mel spectrogram. Note that, when r > 1, the decoder
                outputs r frames of mel spectrogram per step.
            alignments (Variable): Shape(N, B, T_mel // r, T_enc), the alignment
                tensor between the decoder and the encoder, where N means number
                of Attention Layers, T_mel means the length of mel spectrogram,
                r means the outputs per decoder step, T_enc means the encoder
                time steps.
            done (Variable): Shape(B, T_mel // r), probability that the
                outputs should stop.
            decoder_states (Variable): Shape(B, T_mel // r, C_dec), decoder
                hidden states, where C_dec means the channels of decoder states.
        """
        if speaker_embed is not None:
            speaker_embed = F.dropout(
                speaker_embed,
                self.dropout,
                dropout_implementation="upscale_in_train")

        keys, values = encoder_out
        enc_time_steps = keys.shape[1]
        if self.use_memory_mask and lengths is not None:
            mask = gen_mask(lengths, enc_time_steps)
        else:
            mask = None

        if text_positions is not None:
            w = self.key_position_rate
            if self.n_speakers > 1:
                w = w * F.squeeze(self.speaker_proj1(speaker_embed), [-1])
            text_pos_embed = self.embed_keys_positions(text_positions, w)
            keys += text_pos_embed  # (B, T, C)

        if frame_positions is not None:
            w = self.query_position_rate
            if self.n_speakers > 1:
                w = w * F.squeeze(self.speaker_proj2(speaker_embed), [-1])
            frame_pos_embed = self.embed_query_positions(frame_positions, w)
        else:
            frame_pos_embed = None

        # pack multiple frames if necessary
        frames = fold_adjacent_frames(frames, self.r)  # assume (B, T, C) input
        # (B, C, T)
        frames = F.transpose(frames, [0, 2, 1])
        x = frames
        x = F.dropout(x,
                      self.dropout,
                      dropout_implementation="upscale_in_train")
        # Prenet
        for layer in self.prenet:
            if isinstance(layer, Conv1DGLU):
                x = layer(x, speaker_embed)
            else:
                x = layer(x)

        # Convolution & Multi-hop Attention
        alignments = []
        for (conv, attn) in self.conv_attn:
            residual = x
            x = conv(x, speaker_embed)
            if attn is not None:
                x = F.transpose(x, [0, 2, 1])  # (B, T, C)
                if frame_pos_embed is not None:
                    x = x + frame_pos_embed
                x, attn_scores = attn(x, (keys, values), mask)
                alignments.append(attn_scores)
                x = F.transpose(x, [0, 2, 1])  #(B, C, T)
            x = F.scale(residual + x, np.sqrt(0.5))

        alignments = F.stack(alignments)

        decoder_states = x
        x = self.last_conv(x)
        outputs = F.sigmoid(x)
        done = F.sigmoid(self.fc(x))

        outputs = F.transpose(outputs, [0, 2, 1])
        decoder_states = F.transpose(decoder_states, [0, 2, 1])
        done = F.squeeze(done, [1])

        outputs = unfold_adjacent_frames(outputs, self.r)
        decoder_states = unfold_adjacent_frames(decoder_states, self.r)
        return outputs, alignments, done, decoder_states

    @property
    def receptive_field(self):
        """Whole receptive field of the causally convolutional decoder."""
        r = 1
        for conv in self.prenet:
            r += conv.dilation[1] * (conv.filter_size[1] - 1)
        for (conv, _) in self.conv_attn:
            r += conv.dilation[1] * (conv.filter_size[1] - 1)
        return r

    def start_sequence(self):
        for layer in self.prenet:
            if isinstance(layer, Conv1DGLU):
                layer.start_sequence()

        for conv, _ in self.conv_attn:
            if isinstance(conv, Conv1DGLU):
                conv.start_sequence()

    def decode(self,
               encoder_out,
               text_positions,
               speaker_embed=None,
               test_inputs=None):
        self.start_sequence()
        keys, values = encoder_out
        batch_size = keys.shape[0]
        assert batch_size == 1, "now only supports single instance inference"
        mask = None  # no mask because we use single instance decoding

        # no dropout in inference
        if speaker_embed is not None:
            speaker_embed = F.dropout(
                speaker_embed,
                self.dropout,
                dropout_implementation="upscale_in_train")

        # since we use single example inference, there is no text_mask
        if text_positions is not None:
            w = self.key_position_rate
            if self.n_speakers > 1:
                # shape (B, )
                w = w * F.squeeze(self.speaker_proj1(speaker_embed), [-1])
            text_pos_embed = self.embed_keys_positions(text_positions, w)
            keys += text_pos_embed  # (B, T, C)

        # statr decoding
        decoder_states = []  # (B, C, 1) tensors
        mel_outputs = []  # (B, C, 1) tensors
        alignments = []  # (B, 1, T_enc) tensors
        dones = []  # (B, 1, 1) tensors
        last_attended = [None] * len(self.conv_attn)
        for idx, monotonic_attn in enumerate(self.force_monotonic_attention):
            if monotonic_attn:
                last_attended[idx] = 0

        if test_inputs is not None:
            # pack multiple frames if necessary # assume (B, T, C) input
            test_inputs = fold_adjacent_frames(test_inputs, self.r)
            test_inputs = F.transpose(test_inputs, [0, 2, 1])

        initial_input = F.zeros((batch_size, self.mel_dim * self.r, 1),
                                dtype=keys.dtype)

        t = 0  # decoder time step
        while True:
            frame_pos = F.fill_constant((batch_size, 1),
                                        value=t + 1,
                                        dtype="int64")
            w = self.query_position_rate
            if self.n_speakers > 1:
                w = w * F.squeeze(self.speaker_proj2(speaker_embed), [-1])
            # (B, T=1, C)
            frame_pos_embed = self.embed_query_positions(frame_pos, w)

            if test_inputs is not None:
                if t >= test_inputs.shape[-1]:
                    break
                current_input = test_inputs[:, :, t:t + 1]
            else:
                if t > 0:
                    current_input = mel_outputs[-1]  # auto-regressive
                else:
                    current_input = initial_input

            x_t = current_input
            x_t = F.dropout(x_t,
                            self.dropout,
                            dropout_implementation="upscale_in_train")

            # Prenet
            for layer in self.prenet:
                if isinstance(layer, Conv1DGLU):
                    x_t = layer.add_input(x_t, speaker_embed)
                else:
                    x_t = layer(x_t)  # (B, C, T=1)

            step_attn_scores = []
            # causal convolutions + multi-hop attentions
            for i, (conv, attn) in enumerate(self.conv_attn):
                residual = x_t  #(B, C, T=1)
                x_t = conv.add_input(x_t, speaker_embed)
                if attn is not None:
                    x_t = F.transpose(x_t, [0, 2, 1])
                    if frame_pos_embed is not None:
                        x_t += frame_pos_embed
                    x_t, attn_scores = attn(
                        x_t, (keys, values), mask,
                        last_attended[i] if test_inputs is None else None)
                    x_t = F.transpose(x_t, [0, 2, 1])
                    step_attn_scores.append(attn_scores)  #(B, T_dec=1, T_enc)
                    # update last attended when necessary
                    if self.force_monotonic_attention[i]:
                        last_attended[i] = np.argmax(attn_scores.numpy(),
                                                     axis=-1)[0][0]
                x_t = F.scale(residual + x_t, np.sqrt(0.5))
            if len(step_attn_scores):
                # (B, 1, T_enc) again
                average_attn_scores = F.reduce_mean(
                    F.stack(step_attn_scores, 0), 0)
            else:
                average_attn_scores = None

            decoder_state_t = x_t
            x_t = self.last_conv(x_t)

            mel_output_t = F.sigmoid(x_t)
            done_t = F.sigmoid(self.fc(x_t))

            decoder_states.append(decoder_state_t)
            mel_outputs.append(mel_output_t)
            if average_attn_scores is not None:
                alignments.append(average_attn_scores)
            dones.append(done_t)

            t += 1

            if test_inputs is None:
                if F.reduce_min(done_t).numpy(
                )[0] > 0.5 and t > self.min_decoder_steps:
                    break
                elif t > self.max_decoder_steps:
                    break

        # concat results
        mel_outputs = F.concat(mel_outputs, axis=-1)
        decoder_states = F.concat(decoder_states, axis=-1)
        dones = F.concat(dones, axis=-1)
        alignments = F.concat(alignments, axis=1)

        mel_outputs = F.transpose(mel_outputs, [0, 2, 1])
        decoder_states = F.transpose(decoder_states, [0, 2, 1])
        dones = F.squeeze(dones, [1])

        mel_outputs = unfold_adjacent_frames(mel_outputs, self.r)
        decoder_states = unfold_adjacent_frames(decoder_states, self.r)

        return mel_outputs, alignments, dones, decoder_states
