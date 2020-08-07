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
from visualdl import LogWriter
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


def add_scalars(self, main_tag, tag_scalar_dict, step, walltime=None):
    """Add scalars to vdl record file.
    Args:
        main_tag (string): The parent name for the tags
        tag_scalar_dict (dict): Key-value pair storing the tag and corresponding values
        step (int): Step of scalars
        walltime (float): Wall time of scalars.
    Example:
        for index in range(1, 101):
            writer.add_scalar(tag="train/loss", value=index*0.2, step=index)
            writer.add_scalar(tag="train/lr", value=index*0.5, step=index)
    """
    import time
    from visualdl.writer.record_writer import RecordFileWriter
    from visualdl.component.base_component import scalar

    fw_logdir = self.logdir
    walltime = round(time.time()) if walltime is None else walltime
    for tag, value in tag_scalar_dict.items():
        tag = os.path.join(fw_logdir, main_tag, tag)
        if '%' in tag:
            raise RuntimeError("% can't appear in tag!")
        if tag in self._all_writers:
            fw = self._all_writers[tag]
        else:
            fw = RecordFileWriter(
                logdir=tag,
                max_queue_size=self._max_queue,
                flush_secs=self._flush_secs,
                filename_suffix=self._filename_suffix)
            self._all_writers.update({tag: fw})
        fw.add_record(
            scalar(tag=main_tag, value=value, step=step, walltime=walltime))


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

    writer = LogWriter(os.path.join(args.output,
                                    'log')) if local_rank == 0 else None
    writer.add_scalars = add_scalars

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
        shuffle=True).reader

    iterator = iter(tqdm(reader))

    global_step += 1

    while global_step <= cfg['train']['max_iteration']:
        try:
            batch = next(iterator)
        except StopIteration as e:
            iterator = iter(tqdm(reader))
            batch = next(iterator)

        character, mel, mel_input, pos_text, pos_mel, stop_tokens = batch

        mel_pred, postnet_pred, attn_probs, stop_preds, attn_enc, attn_dec = model(
            character, mel_input, pos_text, pos_mel)

        mel_loss = layers.mean(
            layers.abs(layers.elementwise_sub(mel_pred, mel)))
        post_mel_loss = layers.mean(
            layers.abs(layers.elementwise_sub(postnet_pred, mel)))
        loss = mel_loss + post_mel_loss

        stop_loss = cross_entropy(
            stop_preds, stop_tokens, weight=cfg['network']['stop_loss_weight'])
        loss = loss + stop_loss

        if local_rank == 0:
            writer.add_scalars('training_loss', {
                'mel_loss': mel_loss.numpy(),
                'post_mel_loss': post_mel_loss.numpy()
            }, global_step)

            writer.add_scalar('stop_loss', stop_loss.numpy(), global_step)

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
                                'batch_size'] // nranks]) * 255)
                        writer.add_image(
                            'Attention_%d_0' % global_step,
                            x,
                            i * 4 + j)

                for i, prob in enumerate(attn_enc):
                    for j in range(cfg['network']['encoder_num_head']):
                        x = np.uint8(
                            cm.viridis(prob.numpy()[j * cfg['train'][
                                'batch_size'] // nranks]) * 255)
                        writer.add_image(
                            'Attention_enc_%d_0' % global_step,
                            x,
                            i * 4 + j)

                for i, prob in enumerate(attn_dec):
                    for j in range(cfg['network']['decoder_num_head']):
                        x = np.uint8(
                            cm.viridis(prob.numpy()[j * cfg['train'][
                                'batch_size'] // nranks]) * 255)
                        writer.add_image(
                            'Attention_dec_%d_0' % global_step,
                            x,
                            i * 4 + j)

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
                os.path.join(args.output, 'checkpoints'), global_step, model,
                optimizer)
        global_step += 1

    if local_rank == 0:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train TransformerTTS model")
    add_config_options_to_parser(parser)
    args = parser.parse_args()
    # Print the whole config setting.
    pprint(vars(args))
    main(args)
