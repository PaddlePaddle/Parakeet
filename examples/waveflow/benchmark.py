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

import os
import random
from pprint import pprint

import argparse
import numpy as np
import paddle.fluid.dygraph as dg
from paddle import fluid

import utils
from parakeet.utils import io
from waveflow import WaveFlow


def add_options_to_parser(parser):
    parser.add_argument(
        '--model',
        type=str,
        default='waveflow',
        help="general name of the model")
    parser.add_argument(
        '--name', type=str, help="specific name of the training model")
    parser.add_argument(
        '--root', type=str, help="root path of the LJSpeech dataset")

    parser.add_argument(
        '--use_gpu',
        type=utils.str2bool,
        default=True,
        help="option to use gpu training")
    parser.add_argument(
        '--use_fp16',
        type=utils.str2bool,
        default=True,
        help="option to use fp16 for inference")

    parser.add_argument(
        '--iteration',
        type=int,
        default=None,
        help=("which iteration of checkpoint to load, "
              "default to load the latest checkpoint"))
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help="path of the checkpoint to load")


def benchmark(config):
    pprint(vars(config))

    # Get checkpoint directory path.
    run_dir = os.path.join("runs", config.model, config.name)
    checkpoint_dir = os.path.join(run_dir, "checkpoint")

    # Configurate device.
    place = fluid.CUDAPlace(0) if config.use_gpu else fluid.CPUPlace()

    with dg.guard(place):
        # Fix random seed.
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        print("Random Seed: ", seed)

        # Build model.
        model = WaveFlow(config, checkpoint_dir)
        model.build(training=False)

        # Run model inference.
        model.benchmark()


if __name__ == "__main__":
    # Create parser.
    parser = argparse.ArgumentParser(
        description="Synthesize audio using WaveNet model")
    add_options_to_parser(parser)
    utils.add_config_options_to_parser(parser)

    # Parse argument from both command line and yaml config file.
    # For conflicting updates to the same field,
    # the preceding update will be overwritten by the following one.
    config = parser.parse_args()
    config = io.add_yaml_config_to_args(config)
    benchmark(config)
