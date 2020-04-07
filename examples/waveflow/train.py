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
import subprocess
import time
from pprint import pprint

import argparse
import numpy as np
import paddle.fluid.dygraph as dg
from paddle import fluid
from tensorboardX import SummaryWriter

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


def train(config):
    use_gpu = config.use_gpu

    # Get the rank of the current training process.
    rank = dg.parallel.Env().local_rank
    nranks = dg.parallel.Env().nranks
    parallel = nranks > 1

    if rank == 0:
        # Print the whole config setting.
        pprint(vars(config))

    # Make checkpoint directory.
    run_dir = os.path.join("runs", config.model, config.name)
    checkpoint_dir = os.path.join(run_dir, "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Create tensorboard logger.
    tb = SummaryWriter(os.path.join(run_dir, "logs")) \
        if rank == 0 else None

    # Configurate device
    place = fluid.CUDAPlace(rank) if use_gpu else fluid.CPUPlace()

    with dg.guard(place):
        # Fix random seed.
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        print("Random Seed: ", seed)

        # Build model.
        model = WaveFlow(config, checkpoint_dir, parallel, rank, nranks, tb)
        iteration = model.build()

        while iteration < config.max_iterations:
            # Run one single training step.
            model.train_step(iteration)

            iteration += 1

            if iteration % config.test_every == 0:
                # Run validation step.
                model.valid_step(iteration)

            if rank == 0 and iteration % config.save_every == 0:
                # Save parameters.
                model.save(iteration)

    # Close TensorBoard.
    if rank == 0:
        tb.close()


if __name__ == "__main__":
    # Create parser.
    parser = argparse.ArgumentParser(description="Train WaveFlow model")
    #formatter_class='default_argparse')
    add_options_to_parser(parser)
    utils.add_config_options_to_parser(parser)

    # Parse argument from both command line and yaml config file.
    # For conflicting updates to the same field, 
    # the preceding update will be overwritten by the following one.
    config = parser.parse_args()
    config = io.add_yaml_config_to_args(config)
    # Force to use fp32 in model training
    vars(config)["use_fp16"] = False
    train(config)
