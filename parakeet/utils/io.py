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
from paddle.fluid.framework import convert_np_dtype_to_dtype_ as convert_np_dtype


def is_main_process():
    local_rank = dg.parallel.Env().local_rank
    return local_rank == 0


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


def _load_latest_checkpoint(checkpoint_dir):
    """Get the iteration number corresponding to the latest saved checkpoint

    Args:
        checkpoint_dir (str): the directory where checkpoint is saved.

    Returns:
        int: the latest iteration number.
    """
    checkpoint_record = os.path.join(checkpoint_dir, "checkpoint")
    # Create checkpoint index file if not exist.
    if (not os.path.isfile(checkpoint_record)):
        return 0

    # Fetch the latest checkpoint index.
    with open(checkpoint_record, "r") as handle:
        latest_checkpoint = handle.readline().split()[-1]
        iteration = int(latest_checkpoint.split("-")[-1])

    return iteration


def _save_checkpoint(checkpoint_dir, iteration):
    """Save the iteration number of the latest model to be checkpointed.

    Args:
        checkpoint_dir (str): the directory where checkpoint is saved.
        iteration (int): the latest iteration number.

    Returns:
        None
    """
    checkpoint_record = os.path.join(checkpoint_dir, "checkpoint")
    # Update the latest checkpoint index.
    with open(checkpoint_record, "w") as handle:
        handle.write("model_checkpoint_path: step-{}".format(iteration))


def load_parameters(model,
                    optimizer=None,
                    checkpoint_dir=None,
                    iteration=None,
                    checkpoint_path=None):
    """Load a specific model checkpoint from disk. 

    Args:
        model (obj): model to load parameters.
        optimizer (obj, optional): optimizer to load states if needed.
            Defaults to None.
        checkpoint_dir (str, optional): the directory where checkpoint is saved.
        iteration (int, optional): if specified, load the specific checkpoint,
            if not specified, load the latest one. Defaults to None.
        checkpoint_path (str, optional): if specified, load the checkpoint
            stored in the checkpoint_path and the argument 'checkpoint_dir' will 
            be ignored. Defaults to None. 

    Returns:
        iteration (int): number of iterations that the loaded checkpoint has 
            been trained.
    """
    if checkpoint_path is not None:
        iteration = int(os.path.basename(checkpoint_path).split("-")[-1])
    elif checkpoint_dir is not None:
        if iteration is None:
            iteration = _load_latest_checkpoint(checkpoint_dir)
        if iteration == 0:
            return iteration
        checkpoint_path = os.path.join(checkpoint_dir,
                                       "step-{}".format(iteration))
    else:
        raise ValueError(
            "At least one of 'checkpoint_dir' and 'checkpoint_path' should be specified!"
        )

    local_rank = dg.parallel.Env().local_rank
    model_dict, optimizer_dict = dg.load_dygraph(checkpoint_path)

    state_dict = model.state_dict()

    # cast to desired data type, for mixed-precision training/inference.
    for k, v in model_dict.items():
        if k in state_dict and convert_np_dtype(v.dtype) != state_dict[
                k].dtype:
            model_dict[k] = v.astype(state_dict[k].numpy().dtype)

    model.set_dict(model_dict)

    print("[checkpoint] Rank {}: loaded model from {}.pdparams".format(
        local_rank, checkpoint_path))

    if optimizer and optimizer_dict:
        optimizer.set_dict(optimizer_dict)
        print("[checkpoint] Rank {}: loaded optimizer state from {}.pdopt".
              format(local_rank, checkpoint_path))

    return iteration


def save_parameters(checkpoint_dir, iteration, model, optimizer=None):
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
    checkpoint_path = os.path.join(checkpoint_dir, "step-{}".format(iteration))
    model_dict = model.state_dict()
    dg.save_dygraph(model_dict, checkpoint_path)
    print("[checkpoint] Saved model to {}.pdparams".format(checkpoint_path))

    if optimizer:
        opt_dict = optimizer.state_dict()
        dg.save_dygraph(opt_dict, checkpoint_path)
        print("[checkpoint] Saved optimzier state to {}.pdopt".format(
            checkpoint_path))

    _save_checkpoint(checkpoint_dir, iteration)
