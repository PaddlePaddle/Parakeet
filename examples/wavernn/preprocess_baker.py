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

import glob
from multiprocessing import Pool, cpu_count
import pickle
import argparse
import os
import tqdm
import csv
import argparse
import numpy as np
import librosa
from pathlib import Path
import pandas as pd
import soundfile as sf

from paddle.io import Dataset
from parakeet.data import batch_spec, batch_wav
from parakeet.datasets import BakerMetaData
from parakeet.audio import AudioProcessor, LogMagnitude
from config import get_cfg_defaults

from utils.audio import melspectrogram, encode_mu_law, float_2_label


class Transform(object):
    def __init__(self, output_dir: Path, sample_rate, n_fft, win_length, hop_length, n_mels, fmin, fmax,
                 ref_level_db, min_level_db, bits, peak_norm=False, mode='RAW', mu_law=True,):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.peak_norm = peak_norm
        self.ref_level_db = ref_level_db
        self.min_level_db = min_level_db
        self.bits = bits
        self.mode = mode
        self.mu_law = mu_law

        self.wav_dir = output_dir / "wav"
        self.mel_dir = output_dir / "mel"
        self.trimed_dir = output_dir / "trimed_wav"
        self.wav_dir.mkdir(exist_ok=True)
        self.mel_dir.mkdir(exist_ok=True)
        self.trimed_dir.mkdir(exist_ok=True)

        if mode != 'RAW' and mode != 'MOL':
            raise RuntimeError('Unknown mode value - ', self.mode)

    def __call__(self, example):
        wav_path, _, _ = example

        base_name = os.path.splitext(os.path.basename(wav_path))[0]

        wav, _ = librosa.load(wav_path, sr=self.sample_rate)
        peak = np.abs(wav).max()
        if self.peak_norm or peak > 1.0:
            wav /= peak
        mel = melspectrogram(wav, self.sample_rate, self.n_fft, self.n_mels, self.fmin, self.fmax,
                             self.ref_level_db, self.min_level_db, self.hop_length, self.win_length)
        if self.mode == 'RAW':
            if self.mu_law:
                quant = encode_mu_law(wav, mu=2**self.bits)
            else:
                quant = float_2_label(wav, bits=self.bits)
        elif self.mode == 'MOL':
            quant = float_2_label(wav, bits=16)

        mel = mel.astype(np.float32)
        # audio = quant.astype(np.int64)
        audio = quant.astype(np.int32)

        np.save(str(self.wav_dir / base_name), audio)
        np.save(str(self.mel_dir / base_name), mel)

        sf.write(str(self.trimed_dir / (base_name+'.wav')), audio, samplerate=self.sample_rate)

        return base_name, mel.shape[-1], audio.shape[-1]


def create_dataset(config, input_dir, output_dir, n_workers, verbose=True):
    input_dir = Path(input_dir).expanduser()
    '''
        BakerMetaData.records: [filename, normalized text, pinyin]
    '''
    dataset = BakerMetaData(input_dir)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(exist_ok=True)

    transform = \
        Transform(output_dir, config.sample_rate, config.n_fft, config.win_length,
                                 config.hop_length, config.num_mels, config.fmin, config.fmax, config.ref_level_db,
                                 config.min_level_db, config.bits, config.peak_norm, config.mode, config.mu_law)

    file_names = []

    pool = Pool(processes=n_workers)

    for info in tqdm.tqdm(pool.imap(transform, dataset), total=len(dataset)):
        base_name, mel_len, audio_len = info
        file_names.append((base_name, mel_len, audio_len))

    meta_data = pd.DataFrame.from_records(file_names)
    meta_data.to_csv(
        str(output_dir / "metadata.csv"), sep="\t", index=None, header=None)
    print("saved meta data in to {}".format(
        os.path.join(output_dir, "metadata.csv")))

    print("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create dataset")
    parser.add_argument(
        "--config",
        type=str,
        metavar="FILE",
        help="extra config to overwrite the default config")
    parser.add_argument(
        "--input", type=str, help="path of the ljspeech dataset")
    parser.add_argument(
        "--output", type=str, help="path to save output dataset")
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        help="options to overwrite --config file and the default config, passing in KEY VALUE pairs"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print msg")
    parser.add_argument(
        "--num_workers", type=int, default=cpu_count()//2, help="The number of worker threads to use for preprocessing")

    config = get_cfg_defaults()
    # add config.data.mode, which equals config.model.mode
    config.data.mode = config.model.mode
    args = parser.parse_args()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    if args.verbose:
        print(config.data)
        print(args)

    create_dataset(config.data, args.input, args.output, args.num_workers, args.verbose)




