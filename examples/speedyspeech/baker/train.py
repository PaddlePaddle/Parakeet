# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys
import logging
import argparse
import dataclasses
from pathlib import Path

import yaml
import jsonlines
import paddle
import numpy as np
from paddle import nn
from paddle.nn import functional as F
from paddle import distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler
from paddle.optimizer import Adam  # No RAdaom
from paddle.optimizer.lr import StepDecay
from paddle import DataParallel
from visualdl import LogWriter

from parakeet.datasets.data_table import DataTable
from parakeet.models.speedyspeech import SpeedySpeech

from parakeet.training.updater import UpdaterBase
from parakeet.training.trainer import Trainer
from parakeet.training.reporter import report
from parakeet.training import extension
from parakeet.training.extensions.snapshot import Snapshot
from parakeet.training.extensions.visualizer import VisualDL
from parakeet.training.seeding import seed_everything

from batch_fn import collate_baker_examples
from speedyspeech_updater import SpeedySpeechUpdater, SpeedySpeechEvaluator
from config import get_cfg_default


def train_sp(args, config):
    # decides device type and whether to run in parallel
    # setup running environment correctly
    if not paddle.is_compiled_with_cuda:
        paddle.set_device("cpu")
    else:
        paddle.set_device("gpu")
        world_size = paddle.distributed.get_world_size()
        if world_size > 1:
            paddle.distributed.init_parallel_env()

    # set the random seed, it is a must for multiprocess training
    seed_everything(config.seed)

    print(
        f"rank: {dist.get_rank()}, pid: {os.getpid()}, parent_pid: {os.getppid()}",
    )

    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True

    # construct dataset for training and validation
    with jsonlines.open(args.train_metadata, 'r') as reader:
        train_metadata = list(reader)
    train_dataset = DataTable(
        data=train_metadata,
        fields=[
            "phones", "tones", "num_phones", "num_frames", "feats", "durations"
        ],
        converters={"feats": np.load, }, )
    with jsonlines.open(args.dev_metadata, 'r') as reader:
        dev_metadata = list(reader)
    dev_dataset = DataTable(
        data=dev_metadata,
        fields=[
            "phones", "tones", "num_phones", "num_frames", "feats", "durations"
        ],
        converters={"feats": np.load, }, )

    # collate function and dataloader
    train_sampler = DistributedBatchSampler(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=True)
    # dev_sampler = DistributedBatchSampler(dev_dataset,
    #                                       batch_size=config.batch_size,
    #                                       shuffle=False,
    #                                       drop_last=False)
    print("samplers done!")

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_baker_examples,
        num_workers=config.num_workers)
    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=config.batch_size,
        collate_fn=collate_baker_examples,
        num_workers=config.num_workers)
    print("dataloaders done!")

    # batch = collate_baker_examples([train_dataset[i] for i in range(10)])
    # # batch = collate_baker_examples([dev_dataset[i] for i in range(10)])
    # import pdb; pdb.set_trace()
    model = SpeedySpeech(**config["model"])
    if world_size > 1:
        model = DataParallel(model)  # TODO, do not use vocab size from config
    # print(model)
    print("model done!")
    optimizer = Adam(
        0.001,
        parameters=model.parameters(),
        grad_clip=nn.ClipGradByGlobalNorm(5.0))
    print("optimizer done!")

    updater = SpeedySpeechUpdater(
        model=model, optimizer=optimizer, dataloader=train_dataloader)

    output_dir = Path(args.output_dir)
    trainer = Trainer(updater, (config.max_epoch, 'epoch'), output_dir)

    evaluator = SpeedySpeechEvaluator(model, dev_dataloader)

    if dist.get_rank() == 0:
        trainer.extend(evaluator, trigger=(1, "epoch"))
        writer = LogWriter(str(output_dir))
        trainer.extend(VisualDL(writer), trigger=(1, "iteration"))
        trainer.extend(
            Snapshot(max_size=config.num_snapshots), trigger=(1, 'epoch'))
    print(trainer.extensions)
    trainer.run()


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(description="Train a ParallelWaveGAN "
                                     "model with Baker Mandrin TTS dataset.")
    parser.add_argument(
        "--config", type=str, help="config file to overwrite default config")
    parser.add_argument("--train-metadata", type=str, help="training data")
    parser.add_argument("--dev-metadata", type=str, help="dev data")
    parser.add_argument("--output-dir", type=str, help="output dir")
    parser.add_argument(
        "--device", type=str, default="gpu", help="device type to use")
    parser.add_argument(
        "--nprocs", type=int, default=1, help="number of processes")
    parser.add_argument("--verbose", type=int, default=1, help="verbose")

    args = parser.parse_args()
    if args.device == "cpu" and args.nprocs > 1:
        raise RuntimeError("Multiprocess training on CPU is not supported.")
    config = get_cfg_default()
    if args.config:
        config.merge_from_file(args.config)

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(config)
    print(
        f"master see the word size: {dist.get_world_size()}, from pid: {os.getpid()}"
    )

    # dispatch
    if args.nprocs > 1:
        dist.spawn(train_sp, (args, config), nprocs=args.nprocs)
    else:
        train_sp(args, config)


if __name__ == "__main__":
    main()
