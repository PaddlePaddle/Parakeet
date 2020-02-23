import itertools
import os
import time

import argparse
import ruamel.yaml
import numpy as np
import paddle.fluid.dygraph as dg


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


def add_yaml_config(config):
    print(config)
    with open(config.config, 'rt') as f:
        yaml_cfg = ruamel.yaml.safe_load(f)
    cfg_vars = vars(config)
    for k, v in yaml_cfg.items():
        if k in cfg_vars and cfg_vars[k] is not None:
            continue
        cfg_vars[k] = v
    return config


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
