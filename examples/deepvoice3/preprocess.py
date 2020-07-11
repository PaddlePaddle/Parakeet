from __future__ import division
import os
import argparse
from ruamel import yaml
import tqdm
from os.path import join
import csv
import numpy as np
import pandas as pd
import librosa
import logging

from parakeet.data import DatasetMixin


class LJSpeechMetaData(DatasetMixin):
    def __init__(self, root):
        self.root = root
        self._wav_dir = join(root, "wavs")
        csv_path = join(root, "metadata.csv")
        self._table = pd.read_csv(
            csv_path,
            sep="|",
            encoding="utf-8",
            header=None,
            quoting=csv.QUOTE_NONE,
            names=["fname", "raw_text", "normalized_text"])

    def get_example(self, i):
        fname, raw_text, normalized_text = self._table.iloc[i]
        abs_fname = join(self._wav_dir, fname + ".wav")
        return fname, abs_fname, raw_text, normalized_text

    def __len__(self):
        return len(self._table)


class Transform(object):
    def __init__(self, sample_rate, n_fft, hop_length, win_length, n_mels, reduction_factor):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.reduction_factor = reduction_factor

    def __call__(self, fname):
        # wave processing
        audio, _ = librosa.load(fname, sr=self.sample_rate)

        # Pad the data to the right size to have a whole number of timesteps,
        # accounting properly for the model reduction factor.
        frames = audio.size // (self.reduction_factor * self.hop_length) + 1
        # librosa's stft extract frame of n_fft size, so we should pad n_fft // 2 on both sidess
        desired_length = (frames * self.reduction_factor - 1) * self.hop_length + self.n_fft
        pad_amount = (desired_length - audio.size) // 2

        # we pad mannually to control the number of generated frames
        if audio.size % 2 == 0:
            audio = np.pad(audio, (pad_amount, pad_amount), mode='reflect')
        else:
            audio = np.pad(audio, (pad_amount, pad_amount + 1), mode='reflect')

        # STFT
        D = librosa.stft(audio, self.n_fft, self.hop_length, self.win_length, center=False)
        S = np.abs(D)
        S_mel = librosa.feature.melspectrogram(sr=self.sample_rate, S=S, n_mels=self.n_mels, fmax=8000.0)

        # log magnitude
        log_spectrogram = np.log(np.clip(S, a_min=1e-5, a_max=None))
        log_mel_spectrogram = np.log(np.clip(S_mel, a_min=1e-5, a_max=None))
        num_frames = log_spectrogram.shape[-1]
        assert num_frames % self.reduction_factor == 0, "num_frames is wrong"
        return (log_spectrogram.T, log_mel_spectrogram.T, num_frames)


def save(output_path, dataset, transform):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    records = []
    for example in tqdm.tqdm(dataset):
        fname, abs_fname, _, normalized_text = example
        log_spec, log_mel_spec, num_frames = transform(abs_fname)
        records.append((num_frames,
                        fname + "_spec.npy", 
                        fname + "_mel.npy", 
                        normalized_text))
        np.save(join(output_path, fname + "_spec"), log_spec)
        np.save(join(output_path, fname + "_mel"), log_mel_spec)
    meta_data = pd.DataFrame.from_records(records)
    meta_data.to_csv(join(output_path, "metadata.csv"), 
                     quoting=csv.QUOTE_NONE, sep="|", encoding="utf-8",
                     header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess ljspeech dataset and save it.")
    parser.add_argument("--config", type=str, required=True, help="config file")
    parser.add_argument("--input", type=str, required=True, help="data path of the original data")
    parser.add_argument("--output", type=str, required=True, help="path to save the preprocessed dataset")

    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        config = yaml.safe_load(f)
    
    print("========= Command Line Arguments ========")
    for k, v in vars(args).items():
        print("{}: {}".format(k, v))
    print("=========== Configurations ==============")
    for k in ["sample_rate", "n_fft", "win_length", 
              "hop_length", "n_mels", "reduction_factor"]:
        print("{}: {}".format(k, config[k]))

    ljspeech_meta = LJSpeechMetaData(args.input)
    transform = Transform(config["sample_rate"],
                          config["n_fft"],
                          config["hop_length"],
                          config["win_length"],
                          config["n_mels"],
                          config["reduction_factor"])
    save(args.output, ljspeech_meta, transform)

