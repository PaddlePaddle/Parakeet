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

from __future__ import division
import numpy as np

import paddle.fluid.layers as F
import paddle.fluid.initializer as I
import paddle.fluid.dygraph as dg


class DeepVoice3(dg.Layer):
    def __init__(self, encoder, decoder, converter, speaker_embedding,
                 use_decoder_states):
        """Deep Voice 3 TTS model.

        Args:
            encoder (Layer): the encoder.
            decoder (Layer): the decoder.
            converter (Layer): the converter.
            speaker_embedding (Layer): the speaker embedding (for multispeaker cases).
            use_decoder_states (bool): use decoder states instead of predicted mel spectrogram as the input of the converter.
        """
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
        """Compute predicted value in a teacher forcing training manner.

        Args:
            text_sequences (Variable): shape(B, T_enc), dtype: int64, text indices.
            text_positions (Variable): shape(B, T_enc), dtype: int64, positions of text indices.
            valid_lengths (Variable): shape(B, ), dtype: int64, valid lengths of utterances.
            speaker_indices (Variable): shape(B, ), dtype: int64, speaker indices for utterances.
            mel_inputs (Variable): shape(B, T_mel, C_mel), dytpe: int64, ground truth mel spectrogram.
            frame_positions (Variable): shape(B, T_dec), dtype: int64, positions of decoder steps.

        Returns:
            (mel_outputs, linear_outputs, alignments, done)
            mel_outputs (Variable): shape(B, T_mel, C_mel), dtype float32, predicted mel spectrogram.
            mel_outputs (Variable): shape(B, T_mel, C_mel), dtype float32, predicted mel spectrogram.
            alignments (Variable): shape(N, B, T_dec, T_enc), dtype float32, predicted attention.
            done (Variable): shape(B, T_dec), dtype float32, predicted done probability.
            (T_mel: time steps of mel spectrogram, T_lin: time steps of linear spectrogra, T_dec, time steps of decoder, T_enc: time steps of encoder.)
        """
        if hasattr(self, "speaker_embedding"):
            speaker_embed = self.speaker_embedding(speaker_indices)
        else:
            speaker_embed = None

        keys, values = self.encoder(text_sequences, speaker_embed)
        mel_outputs, alignments, done, decoder_states = self.decoder(
            (keys, values), valid_lengths, mel_inputs, text_positions,
            frame_positions, speaker_embed)
        linear_outputs = self.converter(decoder_states
                                        if self.use_decoder_states else
                                        mel_outputs, speaker_embed)
        return mel_outputs, linear_outputs, alignments, done

    def transduce(self, text_sequences, text_positions, speaker_indices=None):
        """Generate output without teacher forcing. Only batch_size = 1 is supported.

        Args:
            text_sequences (Variable): shape(B, T_enc), dtype: int64, text indices.
            text_positions (Variable): shape(B, T_enc), dtype: int64, positions of text indices.
            speaker_indices (Variable): shape(B, ), dtype: int64, speaker indices for utterances.

        Returns:
            (mel_outputs, linear_outputs, alignments, done)
            mel_outputs (Variable): shape(B, T_mel, C_mel), dtype float32, predicted mel spectrogram.
            mel_outputs (Variable): shape(B, T_mel, C_mel), dtype float32, predicted mel spectrogram.
            alignments (Variable): shape(B, T_dec, T_enc), dtype float32, predicted average attention of all attention layers.
            done (Variable): shape(B, T_dec), dtype float32, predicted done probability.
            (T_mel: time steps of mel spectrogram, T_lin: time steps of linear spectrogra, T_dec, time steps of decoder, T_enc: time steps of encoder.)
        """
        if hasattr(self, "speaker_embedding"):
            speaker_embed = self.speaker_embedding(speaker_indices)
        else:
            speaker_embed = None

        keys, values = self.encoder(text_sequences, speaker_embed)
        mel_outputs, alignments, done, decoder_states = self.decoder.decode(
            (keys, values), text_positions, speaker_embed)
        linear_outputs = self.converter(decoder_states
                                        if self.use_decoder_states else
                                        mel_outputs, speaker_embed)
        return mel_outputs, linear_outputs, alignments, done
