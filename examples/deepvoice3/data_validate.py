import os
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from scipy import signal, io

from parakeet.data import DatasetMixin, TransformDataset, FilterDataset
from parakeet.g2p.en import text_to_sequence, sequence_to_text


class LJSpeechMetaData(DatasetMixin):
    def __init__(self, root):
        self.root = Path(root)
        csv_path = self.root.joinpath("train.txt")
        self._table = pd.read_csv(
            csv_path,
            sep="|",
            header=None,
            quoting=csv.QUOTE_NONE,
            names=["lin_spec", "mel_spec", "n_frames", "text"])

    def get_example(self, i):
        lin_spec, mel_spec, n_frames, text = self._table.iloc[i]
        lin_spec = str(self.root.joinpath(lin_spec))
        mel_spec = str(self.root.joinpath(mel_spec))
        return lin_spec, mel_spec, n_frames, text + "\n"

    def __len__(self):
        return len(self._table)


class Transform(object):
    def __init__(self, replace_pronounciation_prob=0.):
        self.replace_pronounciation_prob = replace_pronounciation_prob

    def __call__(self, in_data):
        lin_spec, mel_spec, n_frames, text = in_data

        # text processing
        mix_grapheme_phonemes = text_to_sequence(
            text, self.replace_pronounciation_prob)
        text_length = len(mix_grapheme_phonemes)
        # CAUTION: positions start from 1
        speaker_id = None

        S_norm = np.load(lin_spec).T.astype(np.float32)
        S_mel_norm = np.load(mel_spec).T.astype(np.float32)

        n_frames = S_mel_norm.shape[-1]  # CAUTION: original number of frames

        return (mix_grapheme_phonemes, text_length, speaker_id, S_norm,
                S_mel_norm, n_frames)


class DataCollector(object):
    def __init__(self, downsample_factor=4, r=1):
        self.downsample_factor = int(downsample_factor)
        self.frames_per_step = int(r)
        self._factor = int(downsample_factor * r)
        self._pad_begin = int(r)  # int(downsample_factor * r)

    def __call__(self, examples):
        batch_size = len(examples)

        # lengths
        text_lengths = np.array([example[1]
                                 for example in examples]).astype(np.int64)
        frames = np.array([example[5]
                           for example in examples]).astype(np.int64)

        max_text_length = int(np.max(text_lengths))
        max_frames = int(np.max(frames))
        if max_frames % self._factor != 0:
            max_frames += (self._factor - max_frames % self._factor)
        max_frames += self._factor
        max_decoder_length = max_frames // self._factor

        # pad time sequence
        text_sequences = []
        lin_specs = []
        mel_specs = []
        done_flags = []
        for example in examples:
            (mix_grapheme_phonemes, text_length, speaker_id, S_norm,
             S_mel_norm, num_frames) = example
            text_sequences.append(
                np.pad(mix_grapheme_phonemes,
                       (0, max_text_length - text_length)))
            lin_specs.append(
                np.pad(S_norm,
                       ((0, 0), (self._pad_begin,
                                 max_frames - self._pad_begin - num_frames))))
            mel_specs.append(
                np.pad(S_mel_norm,
                       ((0, 0), (self._pad_begin,
                                 max_frames - self._pad_begin - num_frames))))
            done_flags.append(
                np.pad(np.zeros((int(np.ceil(num_frames // self._factor)), )),
                       (0, max_decoder_length -
                        int(np.ceil(num_frames // self._factor))),
                       constant_values=1))
        text_sequences = np.array(text_sequences).astype(np.int64)
        lin_specs = np.transpose(np.array(lin_specs),
                                 (0, 2, 1)).astype(np.float32)
        mel_specs = np.transpose(np.array(mel_specs),
                                 (0, 2, 1)).astype(np.float32)
        done_flags = np.array(done_flags).astype(np.float32)

        # text positions
        text_mask = (np.arange(1, 1 + max_text_length) <= np.expand_dims(
            text_lengths, -1)).astype(np.int64)
        text_positions = np.arange(1, 1 + max_text_length) * text_mask

        # decoder_positions
        decoder_positions = np.tile(
            np.expand_dims(np.arange(1, 1 + max_decoder_length), 0),
            (batch_size, 1))

        return (text_sequences, text_lengths, text_positions, mel_specs,
                lin_specs, frames, decoder_positions, done_flags)
