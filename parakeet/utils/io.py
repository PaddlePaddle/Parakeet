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
import time

import ruamel.yaml
import numpy as np
import paddle.fluid.dygraph as dg


def add_yaml_config_to_args(config):
    """ Add args in yaml config to the args parsed by argparse. The argument in 
        yaml config will be overwritten by the same argument in argparse if they 
        are both valid.
    
    Args:
        config (args): the args returned by `argparse.ArgumentParser().parse_args()`

    Returns:
        config: the args added yaml config.
    """
    with open(config.config, 'rt') as f:
        yaml_cfg = ruamel.yaml.safe_load(f)
    cfg_vars = vars(config)
    for k, v in yaml_cfg.items():
        if k in cfg_vars and cfg_vars[k] is not None:
            continue
        cfg_vars[k] = v
    return config


def load_latest_checkpoint(checkpoint_dir, rank=0):
    """Get the iteration number corresponding to the latest saved checkpoint

    Args:
        checkpoint_dir (str): the directory where checkpoint is saved.
        rank (int, optional): the rank of the process in multi-process setting.
            Defaults to 0.

    Returns:
        int: the latest iteration number.
    """
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
    """Save the iteration number of the latest model to be checkpointed.

    Args:
        checkpoint_dir (str): the directory where checkpoint is saved.
        iteration (int): the latest iteration number.

    Returns:
        None
    """
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
    # Update the latest checkpoint index.
    with open(checkpoint_path, "w") as handle:
        handle.write("model_checkpoint_path: step-{}".format(iteration))


def load_parameters(checkpoint_dir,
                    rank,
                    model,
                    optimizer=None,
                    iteration=None,
                    file_path=None,
                    dtype="float32"):
    """Load a specific model checkpoint from disk.

    Args:
        checkpoint_dir (str): the directory where checkpoint is saved.
        rank (int): the rank of the process in multi-process setting.
        model (obj): model to load parameters.
        optimizer (obj, optional): optimizer to load states if needed.
            Defaults to None.
        iteration (int, optional): if specified, load the specific checkpoint,
            if not specified, load the latest one. Defaults to None.
        file_path (str, optional): if specified, load the checkpoint
            stored in the file_path. Defaults to None.
        dtype (str, optional): precision of the model parameters.
            Defaults to float32.

    Returns:
        None
    """
    if file_path is None:
        if iteration is None:
            iteration = load_latest_checkpoint(checkpoint_dir, rank)
        if iteration == 0:
            return
        file_path = "{}/step-{}".format(checkpoint_dir, iteration)

    model_dict, optimizer_dict = dg.load_dygraph(file_path)
    if dtype == "float16":
        for k, v in model_dict.items():
            if "conv2d_transpose" in k:
                model_dict[k] = v.astype("float32")
            else:
                model_dict[k] = v.astype(dtype)
    model.set_dict(model_dict)
    print("[checkpoint] Rank {}: loaded model from {}".format(rank, file_path))
    if optimizer and optimizer_dict:
        optimizer.set_dict(optimizer_dict)
        print("[checkpoint] Rank {}: loaded optimizer state from {}".format(
            rank, file_path))


def save_latest_parameters(checkpoint_dir, iteration, model, optimizer=None):
    """Checkpoint the latest trained model parameters.

    Args:
        checkpoint_dir (str): the directory where checkpoint is saved.
        iteration (int): the latest iteration number.
        model (obj): model to be checkpointed.
        optimizer (obj, optional): optimizer to be checkpointed.
            Defaults to None.

    Returns:
        None
    """
    file_path = "{}/step-{}".format(checkpoint_dir, iteration)
    model_dict = model.state_dict()
    dg.save_dygraph(model_dict, file_path)
    print("[checkpoint] Saved model to {}".format(file_path))

    if optimizer:
        opt_dict = optimizer.state_dict()
        dg.save_dygraph(opt_dict, file_path)
        print("[checkpoint] Saved optimzier state to {}".format(file_path))
