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
        self._wav_dir = self.root.joinpath("wavs")
        csv_path = self.root.joinpath("metadata.csv")
        self._table = pd.read_csv(
            csv_path,
            sep="|",
            header=None,
            quoting=csv.QUOTE_NONE,
            names=["fname", "raw_text", "normalized_text"])

    def get_example(self, i):
        fname, raw_text, normalized_text = self._table.iloc[i]
        fname = str(self._wav_dir.joinpath(fname + ".wav"))
        return fname, raw_text, normalized_text

    def __len__(self):
        return len(self._table)


class Transform(object):
    def __init__(self,
                 replace_pronounciation_prob=0.,
                 sample_rate=22050,
                 preemphasis=.97,
                 n_fft=1024,
                 win_length=1024,
                 hop_length=256,
                 fmin=125,
                 fmax=7600,
                 n_mels=80,
                 min_level_db=-100,
                 ref_level_db=20,
                 max_norm=0.999,
                 clip_norm=True):
        self.replace_pronounciation_prob = replace_pronounciation_prob

        self.sample_rate = sample_rate
        self.preemphasis = preemphasis
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.fmin = fmin
        self.fmax = fmax
        self.n_mels = n_mels

        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db
        self.max_norm = max_norm
        self.clip_norm = clip_norm

    def __call__(self, in_data):
        fname, _, normalized_text = in_data

        # text processing
        mix_grapheme_phonemes = text_to_sequence(
            normalized_text, self.replace_pronounciation_prob)
        text_length = len(mix_grapheme_phonemes)
        # CAUTION: positions start from 1
        speaker_id = None

        # wave processing
        wav, _ = librosa.load(fname, sr=self.sample_rate)
        # preemphasis
        y = signal.lfilter([1., -self.preemphasis], [1.], wav)

        # STFT
        D = librosa.stft(y=y,
                         n_fft=self.n_fft,
                         win_length=self.win_length,
                         hop_length=self.hop_length)
        S = np.abs(D)

        # to db and normalize to 0-1
        amplitude_min = np.exp(self.min_level_db / 20 * np.log(10))  # 1e-5
        S_norm = 20 * np.log10(np.maximum(amplitude_min,
                                          S)) - self.ref_level_db
        S_norm = (S_norm - self.min_level_db) / (-self.min_level_db)
        S_norm = self.max_norm * S_norm
        if self.clip_norm:
            S_norm = np.clip(S_norm, 0, self.max_norm)

        # mel scale and to db and normalize to 0-1,
        # CAUTION: pass linear scale S, not dbscaled S
        S_mel = librosa.feature.melspectrogram(S=S,
                                               n_mels=self.n_mels,
                                               fmin=self.fmin,
                                               fmax=self.fmax,
                                               power=1.)
        S_mel = 20 * np.log10(np.maximum(amplitude_min,
                                         S_mel)) - self.ref_level_db
        S_mel_norm = (S_mel - self.min_level_db) / (-self.min_level_db)
        S_mel_norm = self.max_norm * S_mel_norm
        if self.clip_norm:
            S_mel_norm = np.clip(S_mel_norm, 0, self.max_norm)

        # num_frames
        n_frames = S_mel_norm.shape[-1]  # CAUTION: original number of frames

        return (mix_grapheme_phonemes, text_length, speaker_id, S_norm,
                S_mel_norm, n_frames)


class DataCollector(object):
    def __init__(self, downsample_factor=4, r=1):
        self.downsample_factor = int(downsample_factor)
        self.frames_per_step = int(r)
        self._factor = int(downsample_factor * r)
        # CAUTION: small diff here
        self._pad_begin = int(downsample_factor * r)

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
        max_frames += self._pad_begin
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
