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

import argparse
import os
import logging
from pathlib import Path

import jsonlines
import numpy as np
import paddle
from paddle import DataParallel
from paddle import distributed as dist
from paddle import nn
from paddle.io import DataLoader, DistributedBatchSampler
from parakeet.datasets.data_table import DataTable
from parakeet.models.fastspeech2 import FastSpeech2
from parakeet.training.extensions.snapshot import Snapshot
from parakeet.training.extensions.visualizer import VisualDL
from parakeet.training.seeding import seed_everything
from parakeet.training.trainer import Trainer
from visualdl import LogWriter
import yaml

from batch_fn import collate_baker_examples
from config import get_cfg_default
from fastspeech2_updater import FastSpeech2Evaluator
from fastspeech2_updater import FastSpeech2Updater

optim_classes = dict(
    adadelta=paddle.optimizer.Adadelta,
    adagrad=paddle.optimizer.Adagrad,
    adam=paddle.optimizer.Adam,
    adamax=paddle.optimizer.Adamax,
    adamw=paddle.optimizer.AdamW,
    lamb=paddle.optimizer.Lamb,
    momentum=paddle.optimizer.Momentum,
    rmsprop=paddle.optimizer.RMSProp,
    sgd=paddle.optimizer.SGD, )


def build_optimizers(model: nn.Layer, optim='adadelta',
                     learning_rate=0.01) -> paddle.optimizer:
    optim_class = optim_classes.get(optim)
    if optim_class is None:
        raise ValueError(f"must be one of {list(optim_classes)}: {optim}")
    else:
        optim = optim_class(
            parameters=model.parameters(), learning_rate=learning_rate)

    optimizers = optim
    return optimizers


def train_sp(args, config):
    # decides device type and whether to run in parallel
    # setup running environment correctly
    if not paddle.is_compiled_with_cuda():
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
            "text", "text_lengths", "speech", "speech_lengths", "durations",
            "pitch", "energy"
        ],
        converters={"speech": np.load,
                    "pitch": np.load,
                    "energy": np.load}, )
    with jsonlines.open(args.dev_metadata, 'r') as reader:
        dev_metadata = list(reader)
    dev_dataset = DataTable(
        data=dev_metadata,
        fields=[
            "text", "text_lengths", "speech", "speech_lengths", "durations",
            "pitch", "energy"
        ],
        converters={"speech": np.load,
                    "pitch": np.load,
                    "energy": np.load}, )

    # collate function and dataloader

    train_sampler = DistributedBatchSampler(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True)

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

    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)

    odim = config.n_mels
    model = FastSpeech2(idim=vocab_size, odim=odim, **config["model"])
    if world_size > 1:
        model = DataParallel(model)
    print("model done!")

    optimizer = build_optimizers(model, **config["optimizer"])
    print("optimizer done!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    updater = FastSpeech2Updater(
        model=model,
        optimizer=optimizer,
        dataloader=train_dataloader,
        output_dir=output_dir,
        **config["updater"])

    trainer = Trainer(updater, (config.max_epoch, 'epoch'), output_dir)

    evaluator = FastSpeech2Evaluator(
        model, dev_dataloader, output_dir=output_dir, **config["updater"])

    if dist.get_rank() == 0:
        trainer.extend(evaluator, trigger=(1, "epoch"))
        writer = LogWriter(str(output_dir))
        trainer.extend(VisualDL(writer), trigger=(1, "iteration"))
        trainer.extend(
            Snapshot(max_size=config.num_snapshots), trigger=(1, 'epoch'))
    # print(trainer.extensions)
    trainer.run()


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(description="Train a FastSpeech2 "
                                     "model with Baker Mandrin TTS dataset.")
    parser.add_argument(
        "--config", type=str, help="config file to overwrite default config.")
    parser.add_argument("--train-metadata", type=str, help="training data.")
    parser.add_argument("--dev-metadata", type=str, help="dev data.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--device", type=str, default="gpu", help="device type to use.")
    parser.add_argument(
        "--nprocs", type=int, default=1, help="number of processes.")
    parser.add_argument("--verbose", type=int, default=1, help="verbose.")
    parser.add_argument(
        "--phones-dict",
        type=str,
        default="phone_id_map.txt ",
        help="phone vocabulary file.")

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
