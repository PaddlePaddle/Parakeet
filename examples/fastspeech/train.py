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
from parse import add_config_options_to_parser
from pprint import pprint
from ruamel import yaml
from tqdm import tqdm
from collections import OrderedDict
from tensorboardX import SummaryWriter
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers
import paddle.fluid as fluid
from parakeet.models.transformer_tts.transformer_tts import TransformerTTS
from parakeet.models.fastspeech.fastspeech import FastSpeech
from parakeet.models.fastspeech.utils import get_alignment
import sys
sys.path.append("../transformer_tts")
from data import LJSpeechLoader


def load_checkpoint(step, model_path):
    model_dict, opti_dict = fluid.dygraph.load_dygraph(
        os.path.join(model_path, step))
    new_state_dict = OrderedDict()
    for param in model_dict:
        if param.startswith('_layers.'):
            new_state_dict[param[8:]] = model_dict[param]
        else:
            new_state_dict[param] = model_dict[param]
    return new_state_dict, opti_dict


def main(args):
    local_rank = dg.parallel.Env().local_rank if args.use_data_parallel else 0
    nranks = dg.parallel.Env().nranks if args.use_data_parallel else 1

    with open(args.config_path) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    global_step = 0
    place = (fluid.CUDAPlace(dg.parallel.Env().dev_id)
             if args.use_data_parallel else fluid.CUDAPlace(0)
             if args.use_gpu else fluid.CPUPlace())

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    path = os.path.join(args.log_dir, 'fastspeech')

    writer = SummaryWriter(path) if local_rank == 0 else None

    with dg.guard(place):
        with fluid.unique_name.guard():
            transformerTTS = TransformerTTS(cfg)
            model_dict, _ = load_checkpoint(
                str(args.transformer_step),
                os.path.join(args.transtts_path, "transformer"))
            transformerTTS.set_dict(model_dict)
            transformerTTS.eval()

        model = FastSpeech(cfg)
        model.train()
        optimizer = fluid.optimizer.AdamOptimizer(
            learning_rate=dg.NoamDecay(1 / (
                cfg['warm_up_step'] * (args.lr**2)), cfg['warm_up_step']),
            parameter_list=model.parameters())
        reader = LJSpeechLoader(
            cfg, args, nranks, local_rank, shuffle=True).reader()

        if args.checkpoint_path is not None:
            model_dict, opti_dict = load_checkpoint(
                str(args.fastspeech_step),
                os.path.join(args.checkpoint_path, "fastspeech"))
            model.set_dict(model_dict)
            optimizer.set_dict(opti_dict)
            global_step = args.fastspeech_step
            print("load checkpoint!!!")

        if args.use_data_parallel:
            strategy = dg.parallel.prepare_context()
            model = fluid.dygraph.parallel.DataParallel(model, strategy)

        for epoch in range(args.epochs):
            pbar = tqdm(reader)

            for i, data in enumerate(pbar):
                pbar.set_description('Processing at epoch %d' % epoch)
                character, mel, mel_input, pos_text, pos_mel, text_length, mel_lens = data

                _, _, attn_probs, _, _, _ = transformerTTS(
                    character, mel_input, pos_text, pos_mel)
                alignment = dg.to_variable(
                    get_alignment(attn_probs, mel_lens, cfg[
                        'transformer_head'])).astype(np.float32)

                global_step += 1

                #Forward
                result = model(
                    character,
                    pos_text,
                    mel_pos=pos_mel,
                    length_target=alignment)
                mel_output, mel_output_postnet, duration_predictor_output, _, _ = result
                mel_loss = layers.mse_loss(mel_output, mel)
                mel_postnet_loss = layers.mse_loss(mel_output_postnet, mel)
                duration_loss = layers.mean(
                    layers.abs(
                        layers.elementwise_sub(duration_predictor_output,
                                               alignment)))
                total_loss = mel_loss + mel_postnet_loss + duration_loss

                if local_rank == 0:
                    writer.add_scalar('mel_loss',
                                      mel_loss.numpy(), global_step)
                    writer.add_scalar('post_mel_loss',
                                      mel_postnet_loss.numpy(), global_step)
                    writer.add_scalar('duration_loss',
                                      duration_loss.numpy(), global_step)
                    writer.add_scalar('learning_rate',
                                      optimizer._learning_rate.step().numpy(),
                                      global_step)

                if args.use_data_parallel:
                    total_loss = model.scale_loss(total_loss)
                    total_loss.backward()
                    model.apply_collective_grads()
                else:
                    total_loss.backward()
                optimizer.minimize(
                    total_loss,
                    grad_clip=fluid.dygraph_grad_clip.GradClipByGlobalNorm(cfg[
                        'grad_clip_thresh']))
                model.clear_gradients()

                # save checkpoint
                if local_rank == 0 and global_step % args.save_step == 0:
                    if not os.path.exists(args.save_path):
                        os.mkdir(args.save_path)
                    save_path = os.path.join(args.save_path,
                                             'fastspeech/%d' % global_step)
                    dg.save_dygraph(model.state_dict(), save_path)
                    dg.save_dygraph(optimizer.state_dict(), save_path)
        if local_rank == 0:
            writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fastspeech model")
    add_config_options_to_parser(parser)
    args = parser.parse_args()
    # Print the whole config setting.
    pprint(args)
    main(args)
