import numpy as np

import paddle.fluid.layers as F
import paddle.fluid.initializer as I
import paddle.fluid.dygraph as dg


class DeepVoice3(dg.Layer):
    def __init__(self, encoder, decoder, converter, speaker_embedding,
                 use_decoder_states):
        super(DeepVoice3, self).__init__()
        if speaker_embedding is None:
            self.n_speakers = 1
        else:
            self.speaker_embedding = speaker_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.converter = converter
        self.use_decoder_states = use_decoder_states

    def forward(self, text_sequences, text_positions, valid_lengths,
                speaker_indices, mel_inputs, frame_positions):
        if hasattr(self, "speaker_embedding"):
            speaker_embed = self.speaker_embedding(speaker_indices)
        else:
            speaker_embed = None

        keys, values = self.encoder(text_sequences, speaker_embed)
        mel_outputs, alignments, done, decoder_states = self.decoder(
            (keys, values), valid_lengths, mel_inputs, text_positions,
            frame_positions, speaker_embed)
        linear_outputs = self.converter(
            decoder_states if self.use_decoder_states else mel_outputs,
            speaker_embed)
        return mel_outputs, linear_outputs, alignments, done

    def transduce(self, text_sequences, text_positions, speaker_indices=None):
        if hasattr(self, "speaker_embedding"):
            speaker_embed = self.speaker_embedding(speaker_indices)
        else:
            speaker_embed = None

        keys, values = self.encoder(text_sequences, speaker_embed)
        mel_outputs, alignments, done, decoder_states = self.decoder.decode(
            (keys, values), text_positions, speaker_embed)
        linear_outputs = self.converter(
            decoder_states if self.use_decoder_states else mel_outputs,
            speaker_embed)
        return mel_outputs, linear_outputs, alignments, done
