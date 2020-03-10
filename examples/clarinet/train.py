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

from __future__ import division
import os
import sys
import argparse
import ruamel.yaml
import random
from tqdm import tqdm
import pickle
import numpy as np
from tensorboardX import SummaryWriter

import paddle.fluid.dygraph as dg
from paddle import fluid

from parakeet.models.wavenet import WaveNet, UpsampleNet
from parakeet.models.clarinet import STFT, Clarinet, ParallelWaveNet
from parakeet.data import TransformDataset, SliceDataset, RandomSampler, SequentialSampler, DataCargo
from parakeet.utils.layer_tools import summary, freeze

from utils import make_output_tree, valid_model, save_checkpoint, load_checkpoint, load_wavenet
sys.path.append("../wavenet")
from data import LJSpeechMetaData, Transform, DataCollector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train a clarinet model with LJspeech and a trained wavenet model."
    )
    parser.add_argument("--config", type=str, help="path of the config file.")
    parser.add_argument(
        "--device", type=int, default=-1, help="device to use.")
    parser.add_argument(
        "--output",
        type=str,
        default="experiment",
        help="path to save student.")
    parser.add_argument("--data", type=str, help="path of LJspeech dataset.")
    parser.add_argument("--resume", type=str, help="checkpoint to load from.")
    parser.add_argument(
        "--wavenet", type=str, help="wavenet checkpoint to use.")
    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        config = ruamel.yaml.safe_load(f)

    ljspeech_meta = LJSpeechMetaData(args.data)

    data_config = config["data"]
    sample_rate = data_config["sample_rate"]
    n_fft = data_config["n_fft"]
    win_length = data_config["win_length"]
    hop_length = data_config["hop_length"]
    n_mels = data_config["n_mels"]
    train_clip_seconds = data_config["train_clip_seconds"]
    transform = Transform(sample_rate, n_fft, win_length, hop_length, n_mels)
    ljspeech = TransformDataset(ljspeech_meta, transform)

    valid_size = data_config["valid_size"]
    ljspeech_valid = SliceDataset(ljspeech, 0, valid_size)
    ljspeech_train = SliceDataset(ljspeech, valid_size, len(ljspeech))

    teacher_config = config["teacher"]
    n_loop = teacher_config["n_loop"]
    n_layer = teacher_config["n_layer"]
    filter_size = teacher_config["filter_size"]
    context_size = 1 + n_layer * sum([filter_size**i for i in range(n_loop)])
    print("context size is {} samples".format(context_size))
    train_batch_fn = DataCollector(context_size, sample_rate, hop_length,
                                   train_clip_seconds)
    valid_batch_fn = DataCollector(
        context_size, sample_rate, hop_length, train_clip_seconds, valid=True)

    batch_size = data_config["batch_size"]
    train_cargo = DataCargo(
        ljspeech_train,
        train_batch_fn,
        batch_size,
        sampler=RandomSampler(ljspeech_train))

    # only batch=1 for validation is enabled
    valid_cargo = DataCargo(
        ljspeech_valid,
        valid_batch_fn,
        batch_size=1,
        sampler=SequentialSampler(ljspeech_valid))

    make_output_tree(args.output)

    if args.device == -1:
        place = fluid.CPUPlace()
    else:
        place = fluid.CUDAPlace(args.device)

    with dg.guard(place):
        # conditioner(upsampling net)
        conditioner_config = config["conditioner"]
        upsampling_factors = conditioner_config["upsampling_factors"]
        upsample_net = UpsampleNet(upscale_factors=upsampling_factors)
        freeze(upsample_net)

        residual_channels = teacher_config["residual_channels"]
        loss_type = teacher_config["loss_type"]
        output_dim = teacher_config["output_dim"]
        log_scale_min = teacher_config["log_scale_min"]
        assert loss_type == "mog" and output_dim == 3, \
            "the teacher wavenet should be a wavenet with single gaussian output"

        teacher = WaveNet(n_loop, n_layer, residual_channels, output_dim,
                          n_mels, filter_size, loss_type, log_scale_min)
        freeze(teacher)

        student_config = config["student"]
        n_loops = student_config["n_loops"]
        n_layers = student_config["n_layers"]
        student_residual_channels = student_config["residual_channels"]
        student_filter_size = student_config["filter_size"]
        student_log_scale_min = student_config["log_scale_min"]
        student = ParallelWaveNet(n_loops, n_layers, student_residual_channels,
                                  n_mels, student_filter_size)

        stft_config = config["stft"]
        stft = STFT(
            n_fft=stft_config["n_fft"],
            hop_length=stft_config["hop_length"],
            win_length=stft_config["win_length"])

        lmd = config["loss"]["lmd"]
        model = Clarinet(upsample_net, teacher, student, stft,
                         student_log_scale_min, lmd)
        summary(model)

        # optim
        train_config = config["train"]
        learning_rate = train_config["learning_rate"]
        anneal_rate = train_config["anneal_rate"]
        anneal_interval = train_config["anneal_interval"]
        lr_scheduler = dg.ExponentialDecay(
            learning_rate, anneal_interval, anneal_rate, staircase=True)
        optim = fluid.optimizer.Adam(
            lr_scheduler, parameter_list=model.parameters())
        gradiant_max_norm = train_config["gradient_max_norm"]
        clipper = fluid.dygraph_grad_clip.GradClipByGlobalNorm(
            gradiant_max_norm)

        assert args.wavenet or args.resume, "you should load from a trained wavenet or resume training; training without a trained wavenet is not recommended."
        if args.wavenet:
            load_wavenet(model, args.wavenet)

        if args.resume:
            load_checkpoint(model, optim, args.resume)

        # loader
        train_loader = fluid.io.DataLoader.from_generator(
            capacity=10, return_list=True)
        train_loader.set_batch_generator(train_cargo, place)

        valid_loader = fluid.io.DataLoader.from_generator(
            capacity=10, return_list=True)
        valid_loader.set_batch_generator(valid_cargo, place)

        # train
        max_iterations = train_config["max_iterations"]
        checkpoint_interval = train_config["checkpoint_interval"]
        eval_interval = train_config["eval_interval"]
        checkpoint_dir = os.path.join(args.output, "checkpoints")
        state_dir = os.path.join(args.output, "states")
        log_dir = os.path.join(args.output, "log")
        writer = SummaryWriter(log_dir)

        # training loop
        global_step = 1
        global_epoch = 1
        while global_step < max_iterations:
            epoch_loss = 0.
            for j, batch in tqdm(enumerate(train_loader), desc="[train]"):
                audios, mels, audio_starts = batch
                model.train()
                loss_dict = model(
                    audios, mels, audio_starts, clip_kl=global_step > 500)

                writer.add_scalar("learning_rate",
                                  optim._learning_rate.step().numpy()[0],
                                  global_step)
                for k, v in loss_dict.items():
                    writer.add_scalar("loss/{}".format(k),
                                      v.numpy()[0], global_step)

                l = loss_dict["loss"]
                step_loss = l.numpy()[0]
                print("[train] loss: {:<8.6f}".format(step_loss))
                epoch_loss += step_loss

                l.backward()
                optim.minimize(l, grad_clip=clipper)
                optim.clear_gradients()

                if global_step % eval_interval == 0:
                    # evaluate on valid dataset
                    valid_model(model, valid_loader, state_dir, global_step,
                                sample_rate)
                if global_step % checkpoint_interval == 0:
                    save_checkpoint(model, optim, checkpoint_dir, global_step)

                global_step += 1

            # epoch loss
            average_loss = epoch_loss / j
            writer.add_scalar("average_loss", average_loss, global_epoch)
            global_epoch += 1
