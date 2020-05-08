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
import numpy as np
import argparse
import os
import time
import math
from pathlib import Path
from pprint import pprint
from ruamel import yaml
from tqdm import tqdm
from matplotlib import cm
from collections import OrderedDict
from tensorboardX import SummaryWriter
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers
import paddle.fluid as fluid
from parakeet.models.fastspeech.fastspeech import FastSpeech
from parakeet.models.fastspeech.utils import get_alignment
from data import LJSpeechLoader
from parakeet.utils import io


def add_config_options_to_parser(parser):
    parser.add_argument("--config", type=str, help="path of the config file")
    parser.add_argument("--use_gpu", type=int, default=0, help="device to use")
    parser.add_argument("--data", type=str, help="path of LJspeech dataset")
    parser.add_argument(
        "--alignments_path", type=str, help="path of alignments")

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--checkpoint", type=str, help="checkpoint to resume from")
    g.add_argument(
        "--iteration",
        type=int,
        help="the iteration of the checkpoint to load from output directory")

    parser.add_argument(
        "--output",
        type=str,
        default="experiment",
        help="path to save experiment results")


def main(args):
    local_rank = dg.parallel.Env().local_rank
    nranks = dg.parallel.Env().nranks
    parallel = nranks > 1

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    global_step = 0
    place = fluid.CUDAPlace(local_rank) if args.use_gpu else fluid.CPUPlace()
    fluid.enable_dygraph(place)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    writer = SummaryWriter(os.path.join(args.output,
                                        'log')) if local_rank == 0 else None

    model = FastSpeech(cfg['network'], num_mels=cfg['audio']['num_mels'])
    model.train()
    optimizer = fluid.optimizer.AdamOptimizer(
        learning_rate=dg.NoamDecay(1 / (cfg['train']['warm_up_step'] *
                                        (cfg['train']['learning_rate']**2)),
                                   cfg['train']['warm_up_step']),
        parameter_list=model.parameters(),
        grad_clip=fluid.clip.GradientClipByGlobalNorm(cfg['train'][
            'grad_clip_thresh']))
    reader = LJSpeechLoader(
        cfg['audio'],
        place,
        args.data,
        args.alignments_path,
        cfg['train']['batch_size'],
        nranks,
        local_rank,
        shuffle=True).reader()

    # Load parameters.
    global_step = io.load_parameters(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=os.path.join(args.output, 'checkpoints'),
        iteration=args.iteration,
        checkpoint_path=args.checkpoint)
    print("Rank {}: checkpoint loaded.".format(local_rank))

    if parallel:
        strategy = dg.parallel.prepare_context()
        model = fluid.dygraph.parallel.DataParallel(model, strategy)

    for epoch in range(cfg['train']['max_epochs']):
        pbar = tqdm(reader)

        for i, data in enumerate(pbar):
            pbar.set_description('Processing at epoch %d' % epoch)
            (character, mel, pos_text, pos_mel, alignment) = data

            global_step += 1

            #Forward
            result = model(
                character, pos_text, mel_pos=pos_mel, length_target=alignment)
            mel_output, mel_output_postnet, duration_predictor_output, _, _ = result
            mel_loss = layers.mse_loss(mel_output, mel)
            mel_postnet_loss = layers.mse_loss(mel_output_postnet, mel)
            duration_loss = layers.mean(
                layers.abs(
                    layers.elementwise_sub(duration_predictor_output,
                                           alignment)))
            total_loss = mel_loss + mel_postnet_loss + duration_loss

            if local_rank == 0:
                writer.add_scalar('mel_loss', mel_loss.numpy(), global_step)
                writer.add_scalar('post_mel_loss',
                                  mel_postnet_loss.numpy(), global_step)
                writer.add_scalar('duration_loss',
                                  duration_loss.numpy(), global_step)
                writer.add_scalar('learning_rate',
                                  optimizer._learning_rate.step().numpy(),
                                  global_step)

            if parallel:
                total_loss = model.scale_loss(total_loss)
                total_loss.backward()
                model.apply_collective_grads()
            else:
                total_loss.backward()
            optimizer.minimize(total_loss)
            model.clear_gradients()

            # save checkpoint
            if local_rank == 0 and global_step % cfg['train'][
                    'checkpoint_interval'] == 0:
                io.save_parameters(
                    os.path.join(args.output, 'checkpoints'), global_step,
                    model, optimizer)

    if local_rank == 0:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fastspeech model")
    add_config_options_to_parser(parser)
    args = parser.parse_args()
    # Print the whole config setting.
    pprint(vars(args))
    main(args)
