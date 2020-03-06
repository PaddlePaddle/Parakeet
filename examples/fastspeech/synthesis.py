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
from tensorboardX import SummaryWriter
from collections import OrderedDict
import argparse
from parse import add_config_options_to_parser
from pprint import pprint
from ruamel import yaml
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
from parakeet.g2p.en import text_to_sequence
from parakeet import audio
from parakeet.models.fastspeech.fastspeech import FastSpeech
from parakeet.models.transformer_tts.utils import *


def load_checkpoint(step, model_path):
    model_dict, _ = fluid.dygraph.load_dygraph(os.path.join(model_path, step))
    new_state_dict = OrderedDict()
    for param in model_dict:
        if param.startswith('_layers.'):
            new_state_dict[param[8:]] = model_dict[param]
        else:
            new_state_dict[param] = model_dict[param]
    return new_state_dict


def synthesis(text_input, args):
    place = (fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace())

    # tensorboard
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    path = os.path.join(args.log_dir, 'synthesis')

    with open(args.config_path) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    writer = SummaryWriter(path)

    with dg.guard(place):
        model = FastSpeech(cfg)
        model.set_dict(
            load_checkpoint(
                str(args.fastspeech_step),
                os.path.join(args.checkpoint_path, "fastspeech")))
        model.eval()

        text = np.asarray(text_to_sequence(text_input))
        text = np.expand_dims(text, axis=0)
        pos_text = np.arange(1, text.shape[1] + 1)
        pos_text = np.expand_dims(pos_text, axis=0)
        enc_non_pad_mask = get_non_pad_mask(pos_text).astype(np.float32)
        enc_slf_attn_mask = get_attn_key_pad_mask(pos_text,
                                                  text).astype(np.float32)

        text = dg.to_variable(text)
        pos_text = dg.to_variable(pos_text)
        enc_non_pad_mask = dg.to_variable(enc_non_pad_mask)
        enc_slf_attn_mask = dg.to_variable(enc_slf_attn_mask)

        mel_output, mel_output_postnet = model(
            text,
            pos_text,
            alpha=args.alpha,
            enc_non_pad_mask=enc_non_pad_mask,
            enc_slf_attn_mask=enc_slf_attn_mask,
            dec_non_pad_mask=None,
            dec_slf_attn_mask=None)

        _ljspeech_processor = audio.AudioProcessor(
            sample_rate=cfg['audio']['sr'],
            num_mels=cfg['audio']['num_mels'],
            min_level_db=cfg['audio']['min_level_db'],
            ref_level_db=cfg['audio']['ref_level_db'],
            n_fft=cfg['audio']['n_fft'],
            win_length=cfg['audio']['win_length'],
            hop_length=cfg['audio']['hop_length'],
            power=cfg['audio']['power'],
            preemphasis=cfg['audio']['preemphasis'],
            signal_norm=True,
            symmetric_norm=False,
            max_norm=1.,
            mel_fmin=0,
            mel_fmax=None,
            clip_norm=True,
            griffin_lim_iters=60,
            do_trim_silence=False,
            sound_norm=False)

        mel_output_postnet = fluid.layers.transpose(
            fluid.layers.squeeze(mel_output_postnet, [0]), [1, 0])
        wav = _ljspeech_processor.inv_melspectrogram(mel_output_postnet.numpy(
        ))
        writer.add_audio(text_input, wav, 0, cfg['audio']['sr'])
        print("Synthesis completed !!!")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Fastspeech model")
    add_config_options_to_parser(parser)
    args = parser.parse_args()
    synthesis("Transformer model is so fast!", args)
