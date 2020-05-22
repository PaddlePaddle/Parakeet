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
import time
import os
import argparse
import ruamel.yaml
import tqdm
from tensorboardX import SummaryWriter
from paddle import fluid
fluid.require_version('1.8.0')
import paddle.fluid.layers as F
import paddle.fluid.dygraph as dg
from parakeet.utils.io import load_parameters, save_parameters

from data import make_data_loader
from model import make_model, make_criterion, make_optimizer
from utils import make_output_tree, add_options, get_place, Evaluator, StateSaver, make_evaluator, make_state_saver

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Deep Voice 3 model with LJSpeech dataset.")
    add_options(parser)
    args, _ = parser.parse_known_args()

    # only use args.device when training in single process
    # when training with distributed.launch, devices are provided by
    # `--selected_gpus` for distributed.launch
    env = dg.parallel.ParallelEnv()
    device_id = env.dev_id if env.nranks > 1 else args.device
    place = get_place(device_id)
    # start dygraph
    dg.enable_dygraph(place)

    with open(args.config, 'rt') as f:
        config = ruamel.yaml.safe_load(f)

    print("Command Line Args: ")
    for k, v in vars(args).items():
        print("{}: {}".format(k, v))

    data_loader = make_data_loader(args.data, config)
    model = make_model(config)
    if env.nranks > 1:
        strategy = dg.parallel.prepare_context()
        model = dg.DataParallel(model, strategy)
    criterion = make_criterion(config)
    optim = make_optimizer(model, config)

    # generation
    synthesis_config = config["synthesis"]
    power = synthesis_config["power"]
    n_iter = synthesis_config["n_iter"]

    # tensorboard & checkpoint preparation
    output_dir = args.output
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "log")
    state_dir = os.path.join(output_dir, "states")
    eval_dir = os.path.join(output_dir, "eval")
    if env.local_rank == 0:
        make_output_tree(output_dir)
        writer = SummaryWriter(logdir=log_dir)
    else:
        writer = None
    sentences = [
        "Scientists at the CERN laboratory say they have discovered a new particle.",
        "There's a way to measure the acute emotional intelligence that has never gone out of style.",
        "President Trump met with other leaders at the Group of 20 conference.",
        "Generative adversarial network or variational auto-encoder.",
        "Please call Stella.",
        "Some have accepted this as a miracle without any physical explanation.",
    ]
    evaluator = make_evaluator(config, sentences, eval_dir, writer)
    state_saver = make_state_saver(config, state_dir, writer)

    # load parameters and optimizer, and opdate iterations done sofar
    if args.checkpoint is not None:
        iteration = load_parameters(
            model, optim, checkpoint_path=args.checkpoint)
    else:
        iteration = load_parameters(
            model, optim, checkpoint_dir=ckpt_dir, iteration=args.iteration)

    # =========================train=========================
    train_config = config["train"]
    max_iter = train_config["max_iteration"]
    snap_interval = train_config["snap_interval"]
    save_interval = train_config["save_interval"]
    eval_interval = train_config["eval_interval"]

    global_step = iteration + 1
    iterator = iter(tqdm.tqdm(data_loader))
    downsample_factor = config["model"]["downsample_factor"]
    while global_step <= max_iter:
        try:
            batch = next(iterator)
        except StopIteration as e:
            iterator = iter(tqdm.tqdm(data_loader))
            batch = next(iterator)

        model.train()
        (text_sequences, text_lengths, text_positions, mel_specs, lin_specs,
         frames, decoder_positions, done_flags) = batch
        downsampled_mel_specs = F.strided_slice(
            mel_specs,
            axes=[1],
            starts=[0],
            ends=[mel_specs.shape[1]],
            strides=[downsample_factor])
        outputs = model(
            text_sequences,
            text_positions,
            text_lengths,
            None,
            downsampled_mel_specs,
            decoder_positions, )
        # mel_outputs, linear_outputs, alignments, done
        inputs = (downsampled_mel_specs, lin_specs, done_flags, text_lengths,
                  frames)
        losses = criterion(outputs, inputs)

        l = losses["loss"]
        if env.nranks > 1:
            l = model.scale_loss(l)
            l.backward()
            model.apply_collective_grads()
        else:
            l.backward()

        # record learning rate before updating
        if env.local_rank == 0:
            writer.add_scalar("learning_rate",
                              optim._learning_rate.step().numpy(), global_step)
        optim.minimize(l)
        optim.clear_gradients()

        # record step losses
        step_loss = {k: v.numpy()[0] for k, v in losses.items()}

        if env.local_rank == 0:
            tqdm.tqdm.write("[Train] global_step: {}\tloss: {}".format(
                global_step, step_loss["loss"]))
            for k, v in step_loss.items():
                writer.add_scalar(k, v, global_step)

        # train state saving, the first sentence in the batch
        if env.local_rank == 0 and global_step % snap_interval == 0:
            input_specs = (mel_specs, lin_specs)
            state_saver(outputs, input_specs, global_step)

        # evaluation
        if env.local_rank == 0 and global_step % eval_interval == 0:
            evaluator(model, global_step)

        # save checkpoint
        if env.local_rank == 0 and global_step % save_interval == 0:
            save_parameters(ckpt_dir, global_step, model, optim)

        global_step += 1
