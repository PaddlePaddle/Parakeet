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
from tensorboardX import SummaryWriter
import os
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
import argparse
from ruamel import yaml
from pprint import pprint
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers
from data import LJSpeechLoader
from parakeet.models.transformer_tts import Vocoder
from parakeet.utils import io


def add_config_options_to_parser(parser):
    parser.add_argument("--config", type=str, help="path of the config file")
    parser.add_argument("--use_gpu", type=int, default=0, help="device to use")
    parser.add_argument("--data", type=str, help="path of LJspeech dataset")

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--checkpoint", type=str, help="checkpoint to resume from")
    g.add_argument(
        "--iteration",
        type=int,
        help="the iteration of the checkpoint to load from output directory")

    parser.add_argument(
        "--output",
        type=str,
        default="vocoder",
        help="path to save experiment results")


def main(args):
    local_rank = dg.parallel.Env().local_rank
    nranks = dg.parallel.Env().nranks
    parallel = nranks > 1

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    global_step = 0
    place = fluid.CUDAPlace(local_rank) if args.use_gpu else fluid.CPUPlace()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    writer = SummaryWriter(os.path.join(args.output,
                                        'log')) if local_rank == 0 else None

    fluid.enable_dygraph(place)
    model = Vocoder(cfg['train']['batch_size'], cfg['vocoder']['hidden_size'],
                    cfg['audio']['num_mels'], cfg['audio']['n_fft'])

    model.train()
    optimizer = fluid.optimizer.AdamOptimizer(
        learning_rate=dg.NoamDecay(1 / (cfg['train']['warm_up_step'] *
                                        (cfg['train']['learning_rate']**2)),
                                   cfg['train']['warm_up_step']),
        parameter_list=model.parameters(),
        grad_clip=fluid.clip.GradientClipByGlobalNorm(cfg['train'][
            'grad_clip_thresh']))

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

    reader = LJSpeechLoader(
        cfg['audio'],
        place,
        args.data,
        cfg['train']['batch_size'],
        nranks,
        local_rank,
        is_vocoder=True).reader()

    for epoch in range(cfg['train']['max_epochs']):
        pbar = tqdm(reader)
        for i, data in enumerate(pbar):
            pbar.set_description('Processing at epoch %d' % epoch)
            mel, mag = data
            mag = dg.to_variable(mag.numpy())
            mel = dg.to_variable(mel.numpy())
            global_step += 1

            mag_pred = model(mel)
            loss = layers.mean(
                layers.abs(layers.elementwise_sub(mag_pred, mag)))

            if parallel:
                loss = model.scale_loss(loss)
                loss.backward()
                model.apply_collective_grads()
            else:
                loss.backward()
            optimizer.minimize(loss)
            model.clear_gradients()

            if local_rank == 0:
                writer.add_scalars('training_loss', {'loss': loss.numpy(), },
                                   global_step)

            # save checkpoint
            if local_rank == 0 and global_step % cfg['train'][
                    'checkpoint_interval'] == 0:
                io.save_parameters(
                    os.path.join(args.output, 'checkpoints'), global_step,
                    model, optimizer)

    if local_rank == 0:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train vocoder model")
    add_config_options_to_parser(parser)
    args = parser.parse_args()
    # Print the whole config setting.
    pprint(args)
    main(args)
