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

import itertools
import os
import time

import jsonargparse
import numpy as np
import paddle.fluid.dygraph as dg


def add_config_options_to_parser(parser):
    parser.add_argument(
        '--valid_size', type=int, help="size of the valid dataset")
    parser.add_argument(
        '--train_clip_second',
        type=float,
        help="the length of audio clip for training")
    parser.add_argument(
        '--sample_rate', type=int, help="sampling rate of audio data file")
    parser.add_argument(
        '--fft_window_shift',
        type=int,
        help="the shift of fft window for each frame")
    parser.add_argument(
        '--fft_window_size',
        type=int,
        help="the size of fft window for each frame")
    parser.add_argument(
        '--fft_size', type=int, help="the size of fft filter on each frame")
    parser.add_argument(
        '--mel_bands',
        type=int,
        help="the number of mel bands when calculating mel spectrograms")

    parser.add_argument(
        '--seed', type=int, help="seed of random initialization for the model")
    parser.add_argument(
        '--batch_size', type=int, help="batch size for training")
    parser.add_argument(
        '--test_every', type=int, help="test interval during training")
    parser.add_argument(
        '--save_every',
        type=int,
        help="checkpointing interval during training")
    parser.add_argument(
        '--max_iterations', type=int, help="maximum training iterations")

    parser.add_argument(
        '--layers', type=int, help="number of dilated convolution layers")
    parser.add_argument(
        '--kernel_width', type=int, help="dilated convolution kernel width")
    parser.add_argument(
        '--dilation_block', type=list, help="dilated convolution kernel width")
    parser.add_argument('--residual_channels', type=int)
    parser.add_argument('--skip_channels', type=int)
    parser.add_argument(
        '--loss_type', type=str, help="mix-gaussian-pdf or softmax")
    parser.add_argument(
        '--num_channels',
        type=int,
        default=None,
        help="number of channels for softmax output")
    parser.add_argument(
        '--num_mixtures',
        type=int,
        default=None,
        help="number of gaussian mixtures for gaussian output")
    parser.add_argument(
        '--log_scale_min',
        type=float,
        default=None,
        help="minimum clip value of log variance of gaussian output")

    parser.add_argument(
        '--conditioner.filter_sizes',
        type=list,
        help="conv2d tranpose op filter sizes for building conditioner")
    parser.add_argument(
        '--conditioner.upsample_factors',
        type=list,
        help="list of upsample factors for building conditioner")

    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--gradient_max_norm', type=float)
    parser.add_argument(
        '--anneal.every',
        type=int,
        help="step interval for annealing learning rate")
    parser.add_argument('--anneal.rate', type=float)

    parser.add_argument('--config', action=jsonargparse.ActionConfigFile)


def pad_to_size(array, length, pad_with=0.0):
    """
    Pad an array on the first (length) axis to a given length.
    """
    padding = length - array.shape[0]
    assert padding >= 0, "Padding required was less than zero"

    paddings = [(0, 0)] * len(array.shape)
    paddings[0] = (0, padding)

    return np.pad(array, paddings, mode='constant', constant_values=pad_with)


def calculate_context_size(config):
    dilations = list(
        itertools.islice(
            itertools.cycle(config.dilation_block), config.layers))
    config.context_size = sum(dilations) + 1
    print("Context size is", config.context_size)


def load_latest_checkpoint(checkpoint_dir, rank=0):
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
    # Create checkpoint index file if not exist.
    if (not os.path.isfile(checkpoint_path)) and rank == 0:
        with open(checkpoint_path, "w") as handle:
            handle.write("model_checkpoint_path: step-0")

    # Make sure that other process waits until checkpoint file is created
    # by process 0.
    while not os.path.isfile(checkpoint_path):
        time.sleep(1)

    # Fetch the latest checkpoint index.
    with open(checkpoint_path, "r") as handle:
        latest_checkpoint = handle.readline().split()[-1]
        iteration = int(latest_checkpoint.split("-")[-1])

    return iteration


def save_latest_checkpoint(checkpoint_dir, iteration):
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
    # Update the latest checkpoint index.
    with open(checkpoint_path, "w") as handle:
        handle.write("model_checkpoint_path: step-{}".format(iteration))


def load_parameters(checkpoint_dir,
                    rank,
                    model,
                    optimizer=None,
                    iteration=None,
                    file_path=None):
    if file_path is None:
        if iteration is None:
            iteration = load_latest_checkpoint(checkpoint_dir, rank)
        if iteration == 0:
            return
        file_path = "{}/step-{}".format(checkpoint_dir, iteration)

    model_dict, optimizer_dict = dg.load_dygraph(file_path)
    model.set_dict(model_dict)
    print("[checkpoint] Rank {}: loaded model from {}".format(rank, file_path))
    if optimizer and optimizer_dict:
        optimizer.set_dict(optimizer_dict)
        print("[checkpoint] Rank {}: loaded optimizer state from {}".format(
            rank, file_path))


def save_latest_parameters(checkpoint_dir, iteration, model, optimizer=None):
    file_path = "{}/step-{}".format(checkpoint_dir, iteration)
    model_dict = model.state_dict()
    dg.save_dygraph(model_dict, file_path)
    print("[checkpoint] Saved model to {}".format(file_path))

    if optimizer:
        opt_dict = optimizer.state_dict()
        dg.save_dygraph(opt_dict, file_path)
        print("[checkpoint] Saved optimzier state to {}".format(file_path))
