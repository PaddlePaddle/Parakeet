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
import paddle
from paddle import jit
from paddle.static import InputSpec
from paddle import nn
from paddle.nn import functional as F
from paddle import distributed as dist
from yacs.config import CfgNode

from parakeet.datasets.data_table import DataTable
from parakeet.models.speedyspeech import SpeedySpeech, SpeedySpeechInference
from parakeet.models.parallel_wavegan import PWGGenerator, PWGInference
from parakeet.modules.normalizer import ZScore

from frontend import text_analysis


def evaluate(args, speedyspeech_config, pwg_config):
    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True

    # construct dataset for evaluation
    sentences = []
    with open(args.text, 'rt') as f:
        for line in f:
            utt_id, sentence = line.strip().split()
            sentences.append((utt_id, sentence))

    model = SpeedySpeech(**speedyspeech_config["model"])
    model.set_state_dict(
        paddle.load(args.speedyspeech_checkpoint)["main_params"])
    model.eval()

    vocoder = PWGGenerator(**pwg_config["generator_params"])
    vocoder.set_state_dict(paddle.load(args.pwg_params))
    vocoder.remove_weight_norm()
    vocoder.eval()
    print("model done!")

    stat = np.load(args.speedyspeech_stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    speedyspeech_normalizer = ZScore(mu, std)

    stat = np.load(args.pwg_stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    pwg_normalizer = ZScore(mu, std)

    speedyspeech_inference = SpeedySpeechInference(speedyspeech_normalizer,
                                                   model)
    speedyspeech_inference.eval()
    speedyspeech_inference = jit.to_static(
        speedyspeech_inference,
        input_spec=[
            InputSpec(
                [-1], dtype=paddle.int64), InputSpec(
                    [-1], dtype=paddle.int64)
        ])
    paddle.jit.save(speedyspeech_inference,
                    os.path.join(args.inference_dir, "speedyspeech"))
    speedyspeech_inference = paddle.jit.load(
        os.path.join(args.inference_dir, "speedyspeech"))

    pwg_inference = PWGInference(pwg_normalizer, vocoder)
    pwg_inference.eval()
    pwg_inference = jit.to_static(
        pwg_inference,
        input_spec=[InputSpec(
            [-1, 80], dtype=paddle.float32), ])
    paddle.jit.save(pwg_inference, os.path.join(args.inference_dir, "pwg"))
    pwg_inference = paddle.jit.load(os.path.join(args.inference_dir, "pwg"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for utt_id, sentence in sentences:
        phones, tones = text_analysis(sentence)

        with paddle.no_grad():
            wav = pwg_inference(speedyspeech_inference(phones, tones))
        sf.write(
            output_dir / (utt_id + ".wav"),
            wav.numpy(),
            samplerate=speedyspeech_config.sr)
        print(f"{utt_id} done!")


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(
        description="Synthesize with speedyspeech & parallel wavegan.")
    parser.add_argument(
        "--speedyspeech-config",
        type=str,
        help="config file for speedyspeech.")
    parser.add_argument(
        "--speedyspeech-checkpoint",
        type=str,
        help="speedyspeech checkpoint to load.")
    parser.add_argument(
        "--speedyspeech-stat",
        type=str,
        help="mean and standard deviation used to normalize spectrogram when training speedyspeech."
    )
    parser.add_argument(
        "--pwg-config", type=str, help="config file for parallelwavegan.")
    parser.add_argument(
        "--pwg-params",
        type=str,
        help="parallel wavegan generator parameters to load.")
    parser.add_argument(
        "--pwg-stat",
        type=str,
        help="mean and standard deviation used to normalize spectrogram when training speedyspeech."
    )
    parser.add_argument(
        "--text",
        type=str,
        help="text to synthesize, a 'utt_id sentence' pair per line")
    parser.add_argument("--output-dir", type=str, help="output dir")
    parser.add_argument(
        "--inference-dir", type=str, help="dir to save inference models")
    parser.add_argument(
        "--device", type=str, default="gpu", help="device type to use")
    parser.add_argument("--verbose", type=int, default=1, help="verbose")

    args = parser.parse_args()
    with open(args.speedyspeech_config) as f:
        speedyspeech_config = CfgNode(yaml.safe_load(f))
    with open(args.pwg_config) as f:
        pwg_config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(speedyspeech_config)
    print(pwg_config)

    evaluate(args, speedyspeech_config, pwg_config)


if __name__ == "__main__":
    main()
