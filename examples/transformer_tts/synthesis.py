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
from tensorboardX import SummaryWriter
from ruamel import yaml
from pathlib import Path
import argparse
from pprint import pprint
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
from parakeet.g2p.en import text_to_sequence
from parakeet.models.transformer_tts.utils import *
from parakeet import audio
from parakeet.models.transformer_tts import Vocoder
from parakeet.models.transformer_tts import TransformerTTS
from parakeet.utils import io


def add_config_options_to_parser(parser):
    parser.add_argument("--config", type=str, help="path of the config file")
    parser.add_argument("--use_gpu", type=int, default=0, help="device to use")
    parser.add_argument(
        "--max_len",
        type=int,
        default=200,
        help="The max length of audio when synthsis.")

    parser.add_argument(
        "--checkpoint_transformer",
        type=str,
        help="transformer_tts checkpoint to synthesis")
    parser.add_argument(
        "--checkpoint_vocoder",
        type=str,
        help="vocoder checkpoint to synthesis")

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

    writer = SummaryWriter(os.path.join(args.output, 'log'))

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

    with fluid.unique_name.guard():
        model_vocoder = Vocoder(
            cfg['train']['batch_size'], cfg['vocoder']['hidden_size'],
            cfg['audio']['num_mels'], cfg['audio']['n_fft'])
        # Load parameters.
        global_step = io.load_parameters(
            model=model_vocoder, checkpoint_path=args.checkpoint_vocoder)
        model_vocoder.eval()
    # init input
    text = np.asarray(text_to_sequence(text_input))
    text = fluid.layers.unsqueeze(dg.to_variable(text), [0])
    mel_input = dg.to_variable(np.zeros([1, 1, 80])).astype(np.float32)
    pos_text = np.arange(1, text.shape[1] + 1)
    pos_text = fluid.layers.unsqueeze(dg.to_variable(pos_text), [0])

    pbar = tqdm(range(args.max_len))
    for i in pbar:
        pos_mel = np.arange(1, mel_input.shape[1] + 1)
        pos_mel = fluid.layers.unsqueeze(dg.to_variable(pos_mel), [0])
        mel_pred, postnet_pred, attn_probs, stop_preds, attn_enc, attn_dec = model(
            text, mel_input, pos_text, pos_mel)
        mel_input = fluid.layers.concat(
            [mel_input, postnet_pred[:, -1:, :]], axis=1)

    mag_pred = model_vocoder(postnet_pred)

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

    # synthesis with cbhg
    wav = _ljspeech_processor.inv_spectrogram(
        fluid.layers.transpose(fluid.layers.squeeze(mag_pred, [0]), [1, 0])
        .numpy())
    global_step = 0
    for i, prob in enumerate(attn_probs):
        for j in range(4):
            x = np.uint8(cm.viridis(prob.numpy()[j]) * 255)
            writer.add_image(
                'Attention_%d_0' % global_step,
                x,
                i * 4 + j,
                dataformats="HWC")

    writer.add_audio(text_input + '(cbhg)', wav, 0, cfg['audio']['sr'])

    if not os.path.exists(os.path.join(args.output, 'samples')):
        os.mkdir(os.path.join(args.output, 'samples'))
    write(
        os.path.join(os.path.join(args.output, 'samples'), 'cbhg.wav'),
        cfg['audio']['sr'], wav)

    # synthesis with griffin-lim
    wav = _ljspeech_processor.inv_melspectrogram(
        fluid.layers.transpose(
            fluid.layers.squeeze(postnet_pred, [0]), [1, 0]).numpy())
    writer.add_audio(text_input + '(griffin)', wav, 0, cfg['audio']['sr'])

    write(
        os.path.join(os.path.join(args.output, 'samples'), 'griffin.wav'),
        cfg['audio']['sr'], wav)
    print("Synthesis completed !!!")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synthesis model")
    add_config_options_to_parser(parser)
    args = parser.parse_args()
    # Print the whole config setting.
    pprint(vars(args))
    synthesis("Parakeet stands for Paddle PARAllel text-to-speech toolkit.",
              args)
