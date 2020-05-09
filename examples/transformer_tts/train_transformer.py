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
from tqdm import tqdm
from tensorboardX import SummaryWriter
from collections import OrderedDict
import argparse
from pprint import pprint
from ruamel import yaml
from matplotlib import cm
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers
from parakeet.models.transformer_tts.utils import cross_entropy
from data import LJSpeechLoader
from parakeet.models.transformer_tts import TransformerTTS
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

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    writer = SummaryWriter(os.path.join(args.output,
                                        'log')) if local_rank == 0 else None

    fluid.enable_dygraph(place)
    network_cfg = cfg['network']
    model = TransformerTTS(
        network_cfg['embedding_size'], network_cfg['hidden_size'],
        network_cfg['encoder_num_head'], network_cfg['encoder_n_layers'],
        cfg['audio']['num_mels'], network_cfg['outputs_per_step'],
        network_cfg['decoder_num_head'], network_cfg['decoder_n_layers'])

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
        shuffle=True).reader()

    for epoch in range(cfg['train']['max_epochs']):
        pbar = tqdm(reader)
        for i, data in enumerate(pbar):
            pbar.set_description('Processing at epoch %d' % epoch)
            character, mel, mel_input, pos_text, pos_mel = data

            global_step += 1

            mel_pred, postnet_pred, attn_probs, stop_preds, attn_enc, attn_dec = model(
                character, mel_input, pos_text, pos_mel)

            mel_loss = layers.mean(
                layers.abs(layers.elementwise_sub(mel_pred, mel)))
            post_mel_loss = layers.mean(
                layers.abs(layers.elementwise_sub(postnet_pred, mel)))
            loss = mel_loss + post_mel_loss

            # Note: When used stop token loss the learning did not work.
            if cfg['network']['stop_token']:
                label = (pos_mel == 0).astype(np.float32)
                stop_loss = cross_entropy(stop_preds, label)
                loss = loss + stop_loss

            if local_rank == 0:
                writer.add_scalars('training_loss', {
                    'mel_loss': mel_loss.numpy(),
                    'post_mel_loss': post_mel_loss.numpy()
                }, global_step)

                if cfg['network']['stop_token']:
                    writer.add_scalar('stop_loss',
                                      stop_loss.numpy(), global_step)

                if parallel:
                    writer.add_scalars('alphas', {
                        'encoder_alpha': model._layers.encoder.alpha.numpy(),
                        'decoder_alpha': model._layers.decoder.alpha.numpy(),
                    }, global_step)
                else:
                    writer.add_scalars('alphas', {
                        'encoder_alpha': model.encoder.alpha.numpy(),
                        'decoder_alpha': model.decoder.alpha.numpy(),
                    }, global_step)

                writer.add_scalar('learning_rate',
                                  optimizer._learning_rate.step().numpy(),
                                  global_step)

                if global_step % cfg['train']['image_interval'] == 1:
                    for i, prob in enumerate(attn_probs):
                        for j in range(cfg['network']['decoder_num_head']):
                            x = np.uint8(
                                cm.viridis(prob.numpy()[j * cfg['train'][
                                    'batch_size'] // 2]) * 255)
                            writer.add_image(
                                'Attention_%d_0' % global_step,
                                x,
                                i * 4 + j,
                                dataformats="HWC")

                    for i, prob in enumerate(attn_enc):
                        for j in range(cfg['network']['encoder_num_head']):
                            x = np.uint8(
                                cm.viridis(prob.numpy()[j * cfg['train'][
                                    'batch_size'] // 2]) * 255)
                            writer.add_image(
                                'Attention_enc_%d_0' % global_step,
                                x,
                                i * 4 + j,
                                dataformats="HWC")

                    for i, prob in enumerate(attn_dec):
                        for j in range(cfg['network']['decoder_num_head']):
                            x = np.uint8(
                                cm.viridis(prob.numpy()[j * cfg['train'][
                                    'batch_size'] // 2]) * 255)
                            writer.add_image(
                                'Attention_dec_%d_0' % global_step,
                                x,
                                i * 4 + j,
                                dataformats="HWC")

            if parallel:
                loss = model.scale_loss(loss)
                loss.backward()
                model.apply_collective_grads()
            else:
                loss.backward()
            optimizer.minimize(loss)
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
    parser = argparse.ArgumentParser(description="Train TransformerTTS model")
    add_config_options_to_parser(parser)
    args = parser.parse_args()
    # Print the whole config setting.
    pprint(vars(args))
    main(args)
