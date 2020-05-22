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
import os
import sys
import argparse
import ruamel.yaml
import random
from tqdm import tqdm
import pickle
import numpy as np
from tensorboardX import SummaryWriter

import paddle.fluid.dygraph as dg
from paddle import fluid
fluid.require_version('1.8.0')

from parakeet.modules.weight_norm import WeightNormWrapper
from parakeet.models.wavenet import WaveNet, UpsampleNet
from parakeet.models.clarinet import STFT, Clarinet, ParallelWaveNet
from parakeet.data import TransformDataset, SliceDataset, RandomSampler, SequentialSampler, DataCargo
from parakeet.utils.layer_tools import summary, freeze
from parakeet.utils import io

from utils import eval_model
sys.path.append("../wavenet")
from data import LJSpeechMetaData, Transform, DataCollector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize audio files from mel spectrogram in the validation set."
    )
    parser.add_argument("--config", type=str, help="path of the config file")
    parser.add_argument(
        "--device", type=int, default=-1, help="device to use.")
    parser.add_argument("--data", type=str, help="path of LJspeech dataset")

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--checkpoint", type=str, help="checkpoint to resume from")
    g.add_argument(
        "--iteration",
        type=int,
        help="the iteration of the checkpoint to load from output directory")

    parser.add_argument(
        "output",
        type=str,
        default="experiment",
        help="path to save the synthesized audio")

    args = parser.parse_args()

    with open(args.config, 'rt') as f:
        config = ruamel.yaml.safe_load(f)

    if args.device == -1:
        place = fluid.CPUPlace()
    else:
        place = fluid.CUDAPlace(args.device)

    dg.enable_dygraph(place)

    ljspeech_meta = LJSpeechMetaData(args.data)

    data_config = config["data"]
    sample_rate = data_config["sample_rate"]
    n_fft = data_config["n_fft"]
    win_length = data_config["win_length"]
    hop_length = data_config["hop_length"]
    n_mels = data_config["n_mels"]
    train_clip_seconds = data_config["train_clip_seconds"]
    transform = Transform(sample_rate, n_fft, win_length, hop_length, n_mels)
    ljspeech = TransformDataset(ljspeech_meta, transform)

    valid_size = data_config["valid_size"]
    ljspeech_valid = SliceDataset(ljspeech, 0, valid_size)
    ljspeech_train = SliceDataset(ljspeech, valid_size, len(ljspeech))

    teacher_config = config["teacher"]
    n_loop = teacher_config["n_loop"]
    n_layer = teacher_config["n_layer"]
    filter_size = teacher_config["filter_size"]
    context_size = 1 + n_layer * sum([filter_size**i for i in range(n_loop)])
    print("context size is {} samples".format(context_size))
    train_batch_fn = DataCollector(context_size, sample_rate, hop_length,
                                   train_clip_seconds)
    valid_batch_fn = DataCollector(
        context_size, sample_rate, hop_length, train_clip_seconds, valid=True)

    batch_size = data_config["batch_size"]
    train_cargo = DataCargo(
        ljspeech_train,
        train_batch_fn,
        batch_size,
        sampler=RandomSampler(ljspeech_train))

    # only batch=1 for validation is enabled
    valid_cargo = DataCargo(
        ljspeech_valid,
        valid_batch_fn,
        batch_size=1,
        sampler=SequentialSampler(ljspeech_valid))

    # conditioner(upsampling net)
    conditioner_config = config["conditioner"]
    upsampling_factors = conditioner_config["upsampling_factors"]
    upsample_net = UpsampleNet(upscale_factors=upsampling_factors)
    freeze(upsample_net)

    residual_channels = teacher_config["residual_channels"]
    loss_type = teacher_config["loss_type"]
    output_dim = teacher_config["output_dim"]
    log_scale_min = teacher_config["log_scale_min"]
    assert loss_type == "mog" and output_dim == 3, \
        "the teacher wavenet should be a wavenet with single gaussian output"

    teacher = WaveNet(n_loop, n_layer, residual_channels, output_dim, n_mels,
                      filter_size, loss_type, log_scale_min)
    # load & freeze upsample_net & teacher
    freeze(teacher)

    student_config = config["student"]
    n_loops = student_config["n_loops"]
    n_layers = student_config["n_layers"]
    student_residual_channels = student_config["residual_channels"]
    student_filter_size = student_config["filter_size"]
    student_log_scale_min = student_config["log_scale_min"]
    student = ParallelWaveNet(n_loops, n_layers, student_residual_channels,
                              n_mels, student_filter_size)

    stft_config = config["stft"]
    stft = STFT(
        n_fft=stft_config["n_fft"],
        hop_length=stft_config["hop_length"],
        win_length=stft_config["win_length"])

    lmd = config["loss"]["lmd"]
    model = Clarinet(upsample_net, teacher, student, stft,
                     student_log_scale_min, lmd)
    summary(model)

    # load parameters
    if args.checkpoint is not None:
        # load from args.checkpoint
        iteration = io.load_parameters(model, checkpoint_path=args.checkpoint)
    else:
        # load from "args.output/checkpoints"
        checkpoint_dir = os.path.join(args.output, "checkpoints")
        iteration = io.load_parameters(
            model, checkpoint_dir=checkpoint_dir, iteration=args.iteration)
    assert iteration > 0, "A trained checkpoint is needed."

    # make generation fast
    for sublayer in model.sublayers():
        if isinstance(sublayer, WeightNormWrapper):
            sublayer.remove_weight_norm()

    # data loader
    valid_loader = fluid.io.DataLoader.from_generator(
        capacity=10, return_list=True)
    valid_loader.set_batch_generator(valid_cargo, place)

    # the directory to save audio files
    synthesis_dir = os.path.join(args.output, "synthesis")
    if not os.path.exists(synthesis_dir):
        os.makedirs(synthesis_dir)

    eval_model(model, valid_loader, synthesis_dir, iteration, sample_rate)
