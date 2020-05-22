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
import ruamel.yaml
import argparse
import tqdm
from tensorboardX import SummaryWriter
from paddle import fluid
fluid.require_version('1.8.0')
import paddle.fluid.dygraph as dg

from parakeet.data import SliceDataset, TransformDataset, CacheDataset, DataCargo, SequentialSampler, RandomSampler
from parakeet.models.wavenet import UpsampleNet, WaveNet, ConditionalWavenet
from parakeet.utils.layer_tools import summary
from parakeet.utils import io

from data import LJSpeechMetaData, Transform, DataCollector
from utils import make_output_tree, valid_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a WaveNet model with LJSpeech.")
    parser.add_argument(
        "--data", type=str, help="path of the LJspeech dataset")
    parser.add_argument("--config", type=str, help="path of the config file")
    parser.add_argument("--device", type=int, default=-1, help="device to use")

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--checkpoint", type=str, help="checkpoint to resume from")
    g.add_argument(
        "--iteration",
        type=int,
        help="the iteration of the checkpoint to load from output directory")

    parser.add_argument(
        "output", type=str, default="experiment", help="path to save results")

    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        config = ruamel.yaml.safe_load(f)

    if args.device == -1:
        place = fluid.CPUPlace()
    else:
        place = fluid.CUDAPlace(args.device)

    dg.enable_dygraph(place)

    print("Command Line Args: ")
    for k, v in vars(args).items():
        print("{}: {}".format(k, v))

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
    ljspeech_valid = CacheDataset(SliceDataset(ljspeech, 0, valid_size))
    ljspeech_train = CacheDataset(
        SliceDataset(ljspeech, valid_size, len(ljspeech)))

    model_config = config["model"]
    n_loop = model_config["n_loop"]
    n_layer = model_config["n_layer"]
    filter_size = model_config["filter_size"]
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

    model_config = config["model"]
    upsampling_factors = model_config["upsampling_factors"]
    encoder = UpsampleNet(upsampling_factors)

    n_loop = model_config["n_loop"]
    n_layer = model_config["n_layer"]
    residual_channels = model_config["residual_channels"]
    output_dim = model_config["output_dim"]
    loss_type = model_config["loss_type"]
    log_scale_min = model_config["log_scale_min"]
    decoder = WaveNet(n_loop, n_layer, residual_channels, output_dim, n_mels,
                      filter_size, loss_type, log_scale_min)

    model = ConditionalWavenet(encoder, decoder)
    summary(model)

    train_config = config["train"]
    learning_rate = train_config["learning_rate"]
    anneal_rate = train_config["anneal_rate"]
    anneal_interval = train_config["anneal_interval"]
    lr_scheduler = dg.ExponentialDecay(
        learning_rate, anneal_interval, anneal_rate, staircase=True)
    gradiant_max_norm = train_config["gradient_max_norm"]
    optim = fluid.optimizer.Adam(
        lr_scheduler,
        parameter_list=model.parameters(),
        grad_clip=fluid.clip.ClipByGlobalNorm(gradiant_max_norm))

    train_loader = fluid.io.DataLoader.from_generator(
        capacity=10, return_list=True)
    train_loader.set_batch_generator(train_cargo, place)

    valid_loader = fluid.io.DataLoader.from_generator(
        capacity=10, return_list=True)
    valid_loader.set_batch_generator(valid_cargo, place)

    max_iterations = train_config["max_iterations"]
    checkpoint_interval = train_config["checkpoint_interval"]
    snap_interval = train_config["snap_interval"]
    eval_interval = train_config["eval_interval"]
    checkpoint_dir = os.path.join(args.output, "checkpoints")
    log_dir = os.path.join(args.output, "log")
    writer = SummaryWriter(log_dir)

    # load parameters and optimizer, and update iterations done so far
    if args.checkpoint is not None:
        iteration = io.load_parameters(
            model, optim, checkpoint_path=args.checkpoint)
    else:
        iteration = io.load_parameters(
            model,
            optim,
            checkpoint_dir=checkpoint_dir,
            iteration=args.iteration)

    global_step = iteration + 1
    iterator = iter(tqdm.tqdm(train_loader))
    while global_step <= max_iterations:
        try:
            batch = next(iterator)
        except StopIteration as e:
            iterator = iter(tqdm.tqdm(train_loader))
            batch = next(iterator)

        audio_clips, mel_specs, audio_starts = batch

        model.train()
        y_var = model(audio_clips, mel_specs, audio_starts)
        loss_var = model.loss(y_var, audio_clips)
        loss_var.backward()
        loss_np = loss_var.numpy()

        writer.add_scalar("loss", loss_np[0], global_step)
        writer.add_scalar("learning_rate",
                          optim._learning_rate.step().numpy()[0], global_step)
        optim.minimize(loss_var)
        optim.clear_gradients()
        print("global_step: {}\tloss: {:<8.6f}".format(global_step, loss_np[
            0]))

        if global_step % snap_interval == 0:
            valid_model(model, valid_loader, writer, global_step, sample_rate)

        if global_step % checkpoint_interval == 0:
            io.save_parameters(checkpoint_dir, global_step, model, optim)

        global_step += 1
