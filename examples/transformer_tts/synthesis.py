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
from scipy.io.wavfile import write
import numpy as np
from tqdm import tqdm
from matplotlib import cm
from visualdl import LogWriter
from ruamel import yaml
from pathlib import Path
import argparse
from pprint import pprint
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
from parakeet.g2p.en import text_to_sequence
from parakeet.models.transformer_tts.utils import *
from parakeet.models.transformer_tts import TransformerTTS
from parakeet.models.waveflow import WaveFlowModule
from parakeet.modules.weight_norm import WeightNormWrapper
from parakeet.utils import io


def add_config_options_to_parser(parser):
    parser.add_argument("--config", type=str, help="path of the config file")
    parser.add_argument("--use_gpu", type=int, default=0, help="device to use")
    parser.add_argument(
        "--stop_threshold",
        type=float,
        default=0.5,
        help="The threshold of stop token which indicates the time step should stop generate spectrum or not."
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=1000,
        help="The max length of audio when synthsis.")

    parser.add_argument(
        "--checkpoint_transformer",
        type=str,
        help="transformer_tts checkpoint for synthesis")
    parser.add_argument(
        "--vocoder",
        type=str,
        default="griffin-lim",
        choices=['griffin-lim', 'waveflow'],
        help="vocoder method")
    parser.add_argument(
        "--config_vocoder", type=str, help="path of the vocoder config file")
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

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # tensorboard
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    writer = LogWriter(os.path.join(args.output, 'log'))

    fluid.enable_dygraph(place)
    with fluid.unique_name.guard():
        network_cfg = cfg['network']
        model = TransformerTTS(
            network_cfg['embedding_size'], network_cfg['hidden_size'],
            network_cfg['encoder_num_head'], network_cfg['encoder_n_layers'],
            cfg['audio']['num_mels'], network_cfg['outputs_per_step'],
            network_cfg['decoder_num_head'], network_cfg['decoder_n_layers'])
        # Load parameters.
        global_step = io.load_parameters(
            model=model, checkpoint_path=args.checkpoint_transformer)
        model.eval()

    # init input
    text = np.asarray(text_to_sequence(text_input))
    text = fluid.layers.unsqueeze(dg.to_variable(text).astype(np.int64), [0])
    mel_input = dg.to_variable(np.zeros([1, 1, 80])).astype(np.float32)
    pos_text = np.arange(1, text.shape[1] + 1)
    pos_text = fluid.layers.unsqueeze(
        dg.to_variable(pos_text).astype(np.int64), [0])

    for i in range(args.max_len):
        pos_mel = np.arange(1, mel_input.shape[1] + 1)
        pos_mel = fluid.layers.unsqueeze(
            dg.to_variable(pos_mel).astype(np.int64), [0])
        mel_pred, postnet_pred, attn_probs, stop_preds, attn_enc, attn_dec = model(
            text, mel_input, pos_text, pos_mel)
        if stop_preds.numpy()[0, -1] > args.stop_threshold:
            break
        mel_input = fluid.layers.concat(
            [mel_input, postnet_pred[:, -1:, :]], axis=1)
    global_step = 0
    for i, prob in enumerate(attn_probs):
        for j in range(4):
            x = np.uint8(cm.viridis(prob.numpy()[j]) * 255)
            writer.add_image(
                'Attention_%d_0' % global_step,
                x,
                i * 4 + j)

    if args.vocoder == 'griffin-lim':
        #synthesis use griffin-lim
        wav = synthesis_with_griffinlim(postnet_pred, cfg['audio'])
    elif args.vocoder == 'waveflow':
        # synthesis use waveflow
        wav = synthesis_with_waveflow(postnet_pred, args,
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
    # synthesis with griffin-lim
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

    mel_spectrogram = fluid.layers.transpose(
        fluid.layers.squeeze(mel_output, [0]), [1, 0])
    mel_spectrogram = fluid.layers.unsqueeze(mel_spectrogram, [0])

    # Build model.
    waveflow = WaveFlowModule(config)
    io.load_parameters(model=waveflow, checkpoint_path=checkpoint)
    for layer in waveflow.sublayers():
        if isinstance(layer, WeightNormWrapper):
            layer.remove_weight_norm()

    # Run model inference.
    wav = waveflow.synthesize(mel_spectrogram, sigma=config.sigma)
    return wav.numpy()[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synthesis model")
    add_config_options_to_parser(parser)
    args = parser.parse_args()
    # Print the whole config setting.
    pprint(vars(args))
    synthesis(
        "Life was like a box of chocolates, you never know what you're gonna get.",
        args)
