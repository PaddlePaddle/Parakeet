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
import argparse
import ruamel.yaml
import numpy as np
import soundfile as sf

from paddle import fluid
fluid.require_version('1.8.0')
import paddle.fluid.layers as F
import paddle.fluid.dygraph as dg
from tensorboardX import SummaryWriter

from parakeet.g2p import en
from parakeet.modules.weight_norm import WeightNormWrapper
from parakeet.utils.layer_tools import summary
from parakeet.utils import io

from model import make_model
from utils import make_evaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthsize waveform with a checkpoint.")
    parser.add_argument("--config", type=str, help="experiment config")
    parser.add_argument("--device", type=int, default=-1, help="device to use")

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--checkpoint", type=str, help="checkpoint to resume from")
    g.add_argument(
        "--iteration",
        type=int,
        help="the iteration of the checkpoint to load from output directory")

    parser.add_argument("text", type=str, help="text file to synthesize")
    parser.add_argument(
        "output", type=str, help="path to save synthesized audio")

    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        config = ruamel.yaml.safe_load(f)

    print("Command Line Args: ")
    for k, v in vars(args).items():
        print("{}: {}".format(k, v))

    if args.device == -1:
        place = fluid.CPUPlace()
    else:
        place = fluid.CUDAPlace(args.device)

    dg.enable_dygraph(place)

    model = make_model(config)
    checkpoint_dir = os.path.join(args.output, "checkpoints")
    if args.checkpoint is not None:
        iteration = io.load_parameters(model, checkpoint_path=args.checkpoint)
    else:
        iteration = io.load_parameters(
            model, checkpoint_dir=checkpoint_dir, iteration=args.iteration)

    # WARNING: don't forget to remove weight norm to re-compute each wrapped layer's weight
    # removing weight norm also speeds up computation
    for layer in model.sublayers():
        if isinstance(layer, WeightNormWrapper):
            layer.remove_weight_norm()

    synthesis_dir = os.path.join(args.output, "synthesis")
    if not os.path.exists(synthesis_dir):
        os.makedirs(synthesis_dir)

    with open(args.text, "rt", encoding="utf-8") as f:
        lines = f.readlines()
        sentences = [line[:-1] for line in lines]

    evaluator = make_evaluator(config, sentences, synthesis_dir)
    evaluator(model, iteration)
