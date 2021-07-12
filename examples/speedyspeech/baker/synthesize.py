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

import os
import sys
import logging
import argparse
import dataclasses
from pathlib import Path

import yaml
import jsonlines
import paddle
import numpy as np
import soundfile as sf
from paddle import nn
from paddle.nn import functional as F
from paddle import distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler
from paddle.optimizer import Adam  # No RAdaom
from paddle.optimizer.lr import StepDecay
from paddle import DataParallel
from visualdl import LogWriter

from parakeet.datasets.data_table import DataTable
from parakeet.models.speedyspeech import SpeedySpeech
from parakeet.models.parallel_wavegan import PWGGenerator

from parakeet.training.updater import UpdaterBase
from parakeet.training.trainer import Trainer
from parakeet.training.reporter import report
from parakeet.training import extension
from parakeet.training.extensions.snapshot import Snapshot
from parakeet.training.extensions.visualizer import VisualDL
from parakeet.training.seeding import seed_everything

from batch_fn import collate_baker_examples
from speedyspeech_updater import SpeedySpeechUpdater, SpeedySpeechEvaluator
from config import get_cfg_default


def evaluate(args, config):
    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True

    # construct dataset for evaluation
    with jsonlines.open(args.test_metadata, 'r') as reader:
        test_metadata = list(reader)
    test_dataset = DataTable(
        data=test_metadata, fields=["utt_id", "phones", "tones"])

    model = SpeedySpeech(**config["model"])
    model.set_state_dict(paddle.load(args.checkpoint)["main_params"])
    model.eval()
    vocoder_config = yaml.safe_load(
        open("../../parallelwave_gan/baker/conf/default.yaml"))
    vocoder = PWGGenerator(**vocoder_config["generator_params"])
    vocoder.set_state_dict(
        paddle.load("../../parallelwave_gan/baker/converted.pdparams"))
    vocoder.remove_weight_norm()
    vocoder.eval()
    # print(model)
    print("model done!")

    stat = np.load("../../speedyspeech/baker/dump/train/stats.npy")
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)

    stat2 = np.load("../../parallelwave_gan/baker/dump/train/stats.npy")
    mu2, std2 = stat2
    mu2 = paddle.to_tensor(mu2)
    std2 = paddle.to_tensor(std2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for datum in test_dataset:
        utt_id = datum["utt_id"]
        phones = paddle.to_tensor(datum["phones"])
        tones = paddle.to_tensor(datum["tones"])

        mel, _ = model.inference(phones, tones)
        mel = mel * std + mu
        mel = (mel - mu2) / std2

        wav = vocoder.inference(mel)
        sf.write(
            output_dir / (utt_id + ".wav"), wav.numpy(), samplerate=config.sr)
        print(f"{utt_id} done!")


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(description="Train a ParallelWaveGAN "
                                     "model with Baker Mandrin TTS dataset.")
    parser.add_argument(
        "--config", type=str, help="config file to overwrite default config")
    parser.add_argument("--checkpoint", type=str, help="checkpoint to load.")
    parser.add_argument("--test-metadata", type=str, help="training data")
    parser.add_argument("--output-dir", type=str, help="output dir")
    parser.add_argument(
        "--device", type=str, default="gpu", help="device type to use")
    parser.add_argument("--verbose", type=int, default=1, help="verbose")

    args = parser.parse_args()
    config = get_cfg_default()
    if args.config:
        config.merge_from_file(args.config)

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(config)

    evaluate(args, config)


if __name__ == "__main__":
    main()
