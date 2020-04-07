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

import argparse


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def add_config_options_to_parser(parser):
    parser.add_argument(
        '--valid_size', type=int, help="size of the valid dataset")
    parser.add_argument(
        '--segment_length',
        type=int,
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
        '--mel_fmin',
        type=float,
        help="lowest frequency in calculating mel spectrograms")
    parser.add_argument(
        '--mel_fmax',
        type=float,
        help="highest frequency in calculating mel spectrograms")

    parser.add_argument(
        '--seed', type=int, help="seed of random initialization for the model")
    parser.add_argument('--learning_rate', type=float)
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
        '--sigma',
        type=float,
        help="standard deviation of the latent Gaussian variable")
    parser.add_argument('--n_flows', type=int, help="number of flows")
    parser.add_argument(
        '--n_group',
        type=int,
        help="number of adjacent audio samples to squeeze into one column")
    parser.add_argument(
        '--n_layers',
        type=int,
        help="number of conv2d layer in one wavenet-like flow architecture")
    parser.add_argument(
        '--n_channels', type=int, help="number of residual channels in flow")
    parser.add_argument(
        '--kernel_h',
        type=int,
        help="height of the kernel in the conv2d layer")
    parser.add_argument(
        '--kernel_w', type=int, help="width of the kernel in the conv2d layer")

    parser.add_argument('--config', type=str, help="Path to the config file.")
