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

import jsonlines
import numpy as np
import paddle
import yaml
from paddle import DataParallel
from paddle import distributed as dist
from paddle import nn
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from paddle.optimizer import Adam  # No RAdaom
from paddle.optimizer.lr import StepDecay
from parakeet.datasets.data_table import DataTable
from parakeet.models.parallel_wavegan import PWGGenerator
from parakeet.models.parallel_wavegan import PWGDiscriminator
from parakeet.modules.stft_loss import MultiResolutionSTFTLoss
from parakeet.training.extensions.snapshot import Snapshot
from parakeet.training.extensions.visualizer import VisualDL
from parakeet.training.seeding import seed_everything
from parakeet.training.trainer import Trainer
from pathlib import Path
from visualdl import LogWriter

from batch_fn import Clip
from config import get_cfg_default
from pwg_updater import PWGUpdater
from pwg_updater import PWGEvaluator


def train_sp(args, config):
    # decides device type and whether to run in parallel
    # setup running environment correctly
    world_size = paddle.distributed.get_world_size()
    if not paddle.is_compiled_with_cuda():
        paddle.set_device("cpu")
    else:
        paddle.set_device("gpu")
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
        fields=["wave", "feats"],
        converters={
            "wave": np.load,
            "feats": np.load,
        }, )
    with jsonlines.open(args.dev_metadata, 'r') as reader:
        dev_metadata = list(reader)
    dev_dataset = DataTable(
        data=dev_metadata,
        fields=["wave", "feats"],
        converters={
            "wave": np.load,
            "feats": np.load,
        }, )

    # collate function and dataloader
    train_sampler = DistributedBatchSampler(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True)
    dev_sampler = DistributedBatchSampler(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False)
    print("samplers done!")

    train_batch_fn = Clip(
        batch_max_steps=config.batch_max_steps,
        hop_size=config.hop_length,
        aux_context_window=config.generator_params.aux_context_window)

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=train_batch_fn,
        num_workers=config.num_workers)

    dev_dataloader = DataLoader(
        dev_dataset,
        batch_sampler=dev_sampler,
        collate_fn=train_batch_fn,
        num_workers=config.num_workers)
    print("dataloaders done!")

    generator = PWGGenerator(**config["generator_params"])
    discriminator = PWGDiscriminator(**config["discriminator_params"])
    if world_size > 1:
        generator = DataParallel(generator)
        discriminator = DataParallel(discriminator)
    print("models done!")

    criterion_stft = MultiResolutionSTFTLoss(**config["stft_loss_params"])
    criterion_mse = nn.MSELoss()
    print("criterions done!")

    lr_schedule_g = StepDecay(**config["generator_scheduler_params"])
    gradient_clip_g = nn.ClipGradByGlobalNorm(config["generator_grad_norm"])
    optimizer_g = Adam(
        learning_rate=lr_schedule_g,
        grad_clip=gradient_clip_g,
        parameters=generator.parameters(),
        **config["generator_optimizer_params"])
    lr_schedule_d = StepDecay(**config["discriminator_scheduler_params"])
    gradient_clip_d = nn.ClipGradByGlobalNorm(config["discriminator_grad_norm"])
    optimizer_d = Adam(
        learning_rate=lr_schedule_d,
        grad_clip=gradient_clip_d,
        parameters=discriminator.parameters(),
        **config["discriminator_optimizer_params"])
    print("optimizers done!")

    output_dir = Path(args.output_dir)
    if dist.get_rank() == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.yaml", 'wt') as f:
            f.write(config.dump(default_flow_style=None))

    updater = PWGUpdater(
        models={
            "generator": generator,
            "discriminator": discriminator,
        },
        optimizers={
            "generator": optimizer_g,
            "discriminator": optimizer_d,
        },
        criterions={
            "stft": criterion_stft,
            "mse": criterion_mse,
        },
        schedulers={
            "generator": lr_schedule_g,
            "discriminator": lr_schedule_d,
        },
        dataloader=train_dataloader,
        discriminator_train_start_steps=config.discriminator_train_start_steps,
        lambda_adv=config.lambda_adv,
        output_dir=output_dir)

    evaluator = PWGEvaluator(
        models={
            "generator": generator,
            "discriminator": discriminator,
        },
        criterions={
            "stft": criterion_stft,
            "mse": criterion_mse,
        },
        dataloader=dev_dataloader,
        lambda_adv=config.lambda_adv,
        output_dir=output_dir)
    trainer = Trainer(
        updater,
        stop_trigger=(config.train_max_steps, "iteration"),
        out=output_dir, )

    if dist.get_rank() == 0:
        trainer.extend(
            evaluator, trigger=(config.eval_interval_steps, 'iteration'))
        writer = LogWriter(str(trainer.out))
        trainer.extend(VisualDL(writer), trigger=(1, 'iteration'))
        trainer.extend(
            Snapshot(max_size=config.num_snapshots),
            trigger=(config.save_interval_steps, 'iteration'))

    # print(trainer.extensions.keys())
    print("Trainer Done!")
    trainer.run()


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(description="Train a ParallelWaveGAN "
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
