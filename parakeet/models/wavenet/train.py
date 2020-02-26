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

import jsonargparse
import numpy as np
import paddle.fluid.dygraph as dg
from paddle import fluid
from tensorboardX import SummaryWriter

import slurm
import utils
from wavenet import WaveNet

MAXIMUM_SAVE_TIME = 10 * 60


def add_options_to_parser(parser):
    parser.add_argument(
        '--model',
        type=str,
        default='wavenet',
        help="general name of the model")
    parser.add_argument(
        '--name', type=str, help="specific name of the training model")
    parser.add_argument(
        '--root', type=str, help="root path of the LJSpeech dataset")

    parser.add_argument(
        '--parallel',
        type=bool,
        default=True,
        help="option to use data parallel training")
    parser.add_argument(
        '--use_gpu',
        type=bool,
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
    parser.add_argument(
        '--slurm',
        type=bool,
        default=False,
        help="whether you are using slurm to submit training jobs")


def train(config):
    use_gpu = config.use_gpu
    parallel = config.parallel if use_gpu else False

    # Get the rank of the current training process.
    rank = dg.parallel.Env().local_rank if parallel else 0
    nranks = dg.parallel.Env().nranks if parallel else 1

    if rank == 0:
        # Print the whole config setting.
        pprint(jsonargparse.namespace_to_dict(config))

    # Make checkpoint directory.
    run_dir = os.path.join("runs", config.model, config.name)
    checkpoint_dir = os.path.join(run_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

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
        model = WaveNet(config, checkpoint_dir, parallel, rank, nranks, tb)
        model.build()

        # Obtain the current iteration.
        if config.checkpoint is None:
            if config.iteration is None:
                iteration = utils.load_latest_checkpoint(checkpoint_dir, rank)
            else:
                iteration = config.iteration
        else:
            iteration = int(config.checkpoint.split('/')[-1].split('-')[-1])

        # Get restart command if using slurm.
        if config.slurm:
            resume_command, death_time = slurm.restart_command()
            if rank == 0:
                print("Restart command:", " ".join(resume_command))
        done = False

        while iteration < config.max_iterations:
            # Run one single training step.
            model.train_step(iteration)

            iteration += 1

            if iteration % config.test_every == 0:
                # Run validation step.
                model.valid_step(iteration)

            # Check whether reaching the time limit.
            if config.slurm:
                done = (death_time is not None and
                        death_time - time.time() < MAXIMUM_SAVE_TIME)

            if rank == 0 and done:
                print("Saving progress before exiting.")
                model.save(iteration)

                print("Running restart command:", " ".join(resume_command))
                # Submit restart command.
                subprocess.check_call(resume_command)
                break

            if rank == 0 and iteration % config.save_every == 0:
                # Save parameters.
                model.save(iteration)

    # Close TensorBoard.
    if rank == 0:
        tb.close()


if __name__ == "__main__":
    # Create parser.
    parser = jsonargparse.ArgumentParser(
        description="Train WaveNet model", formatter_class='default_argparse')
    add_options_to_parser(parser)
    utils.add_config_options_to_parser(parser)

    # Parse argument from both command line and yaml config file.
    # For conflicting updates to the same field, 
    # the preceding update will be overwritten by the following one.
    config = parser.parse_args()
    train(config)
