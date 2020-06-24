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
from scipy.io.wavfile import write
from collections import OrderedDict
import argparse
from pprint import pprint
from ruamel import yaml
from matplotlib import cm
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
from parakeet.g2p.en import text_to_sequence
from parakeet import audio
from parakeet.models.fastspeech.fastspeech import FastSpeech
from parakeet.models.transformer_tts.utils import *
from parakeet.models.wavenet import WaveNet, UpsampleNet
from parakeet.models.clarinet import STFT, Clarinet, ParallelWaveNet
from parakeet.modules import weight_norm
from parakeet.models.waveflow import WaveFlowModule
from parakeet.utils.layer_tools import freeze
from parakeet.utils import io


def add_config_options_to_parser(parser):
    parser.add_argument("--config", type=str, help="path of the config file")
    parser.add_argument(
        "--vocoder",
        type=str,
        default="griffin-lim",
        choices=['griffin-lim', 'waveflow'],
        help="vocoder method")
    parser.add_argument(
        "--config_vocoder", type=str, help="path of the vocoder config file")
    parser.add_argument("--use_gpu", type=int, default=0, help="device to use")
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
        help="determine the length of the expanded sequence mel, controlling the voice speed."
    )

    parser.add_argument(
        "--checkpoint", type=str, help="fastspeech checkpoint for synthesis")
    parser.add_argument(
        "--checkpoint_vocoder",
        type=str,
        help="vocoder checkpoint for synthesis")

    parser.add_argument(
        "--output",
        type=str,
        default="synthesis",
        help="path to save experiment results")


def synthesis(text_input, args):
    local_rank = dg.parallel.Env().local_rank
    place = (fluid.CUDAPlace(local_rank) if args.use_gpu else fluid.CPUPlace())
    fluid.enable_dygraph(place)

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # tensorboard
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    writer = SummaryWriter(os.path.join(args.output, 'log'))

    model = FastSpeech(cfg['network'], num_mels=cfg['audio']['num_mels'])
    # Load parameters.
    global_step = io.load_parameters(
        model=model, checkpoint_path=args.checkpoint)
    model.eval()

    text = np.asarray(text_to_sequence(text_input))
    text = np.expand_dims(text, axis=0)
    pos_text = np.arange(1, text.shape[1] + 1)
    pos_text = np.expand_dims(pos_text, axis=0)

    text = dg.to_variable(text).astype(np.int64)
    pos_text = dg.to_variable(pos_text).astype(np.int64)

    _, mel_output_postnet = model(text, pos_text, alpha=args.alpha)

    if args.vocoder == 'griffin-lim':
        #synthesis use griffin-lim
        wav = synthesis_with_griffinlim(mel_output_postnet, cfg['audio'])
    elif args.vocoder == 'waveflow':
        wav = synthesis_with_waveflow(mel_output_postnet, args,
                                      args.checkpoint_vocoder, place)
    else:
        print(
            'vocoder error, we only support griffinlim and waveflow, but recevied %s.'
            % args.vocoder)

    writer.add_audio(text_input + '(' + args.vocoder + ')', wav, 0,
                     cfg['audio']['sr'])
    if not os.path.exists(os.path.join(args.output, 'samples')):
        os.mkdir(os.path.join(args.output, 'samples'))
    write(
        os.path.join(
            os.path.join(args.output, 'samples'), args.vocoder + '.wav'),
        cfg['audio']['sr'], wav)
    print("Synthesis completed !!!")
    writer.close()


def synthesis_with_griffinlim(mel_output, cfg):
    mel_output = fluid.layers.transpose(
        fluid.layers.squeeze(mel_output, [0]), [1, 0])
    mel_output = np.exp(mel_output.numpy())
    basis = librosa.filters.mel(cfg['sr'],
                                cfg['n_fft'],
                                cfg['num_mels'],
                                fmin=cfg['fmin'],
                                fmax=cfg['fmax'])
    inv_basis = np.linalg.pinv(basis)
    spec = np.maximum(1e-10, np.dot(inv_basis, mel_output))

    wav = librosa.core.griffinlim(
        spec**cfg['power'],
        hop_length=cfg['hop_length'],
        win_length=cfg['win_length'])

    return wav


def synthesis_with_waveflow(mel_output, args, checkpoint, place):

    fluid.enable_dygraph(place)
    args.config = args.config_vocoder
    args.use_fp16 = False
    config = io.add_yaml_config_to_args(args)

    mel_spectrogram = fluid.layers.transpose(mel_output, [0, 2, 1])

    # Build model.
    waveflow = WaveFlowModule(config)
    io.load_parameters(model=waveflow, checkpoint_path=checkpoint)
    for layer in waveflow.sublayers():
        if isinstance(layer, weight_norm.WeightNormWrapper):
            layer.remove_weight_norm()

    # Run model inference.
    wav = waveflow.synthesize(mel_spectrogram, sigma=config.sigma)
    return wav.numpy()[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synthesis model")
    add_config_options_to_parser(parser)
    args = parser.parse_args()
    pprint(vars(args))
    synthesis(
        "Don't argue with the people of strong determination, because they may change the fact!",
        args)
