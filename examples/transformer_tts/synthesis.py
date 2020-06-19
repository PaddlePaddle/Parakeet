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
from parakeet.modules import weight_norm
from parakeet.models.waveflow import WaveFlowModule
from parakeet.modules.weight_norm import WeightNormWrapper
from parakeet.models.wavenet import UpsampleNet, WaveNet, ConditionalWavenet
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
        "--vocoder",
        type=str,
        default="griffinlim",
        choices=['griffinlim', 'wavenet', 'waveflow'],
        help="vocoder method")
    parser.add_argument(
        "--config_vocoder", type=str, help="path of the vocoder config file")
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

    # init input
    text = np.asarray(text_to_sequence(text_input))
    text = fluid.layers.unsqueeze(dg.to_variable(text).astype(np.int64), [0])
    mel_input = dg.to_variable(np.zeros([1, 1, 80])).astype(np.float32)
    pos_text = np.arange(1, text.shape[1] + 1)
    pos_text = fluid.layers.unsqueeze(
        dg.to_variable(pos_text).astype(np.int64), [0])

    pbar = tqdm(range(args.max_len))
    for i in pbar:
        pos_mel = np.arange(1, mel_input.shape[1] + 1)
        pos_mel = fluid.layers.unsqueeze(
            dg.to_variable(pos_mel).astype(np.int64), [0])
        mel_pred, postnet_pred, attn_probs, stop_preds, attn_enc, attn_dec = model(
            text, mel_input, pos_text, pos_mel)
        mel_input = fluid.layers.concat(
            [mel_input, postnet_pred[:, -1:, :]], axis=1)
    global_step = 0
    for i, prob in enumerate(attn_probs):
        for j in range(4):
            x = np.uint8(cm.viridis(prob.numpy()[j]) * 255)
            writer.add_image(
                'Attention_%d_0' % global_step,
                x,
                i * 4 + j,
                dataformats="HWC")

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
        mel_fmax=8000,
        clip_norm=True,
        griffin_lim_iters=60,
        do_trim_silence=False,
        sound_norm=False)

    if args.vocoder == 'griffinlim':
        #synthesis use griffin-lim
        wav = synthesis_with_griffinlim(postnet_pred, _ljspeech_processor)
    elif args.vocoder == 'wavenet':
        # synthesis use wavenet
        wav = synthesis_with_wavenet(postnet_pred, args)
    elif args.vocoder == 'waveflow':
        # synthesis use waveflow
        wav = synthesis_with_waveflow(postnet_pred, args,
                                      args.checkpoint_vocoder,
                                      _ljspeech_processor, place)
    else:
        print(
            'vocoder error, we only support griffinlim, cbhg and waveflow, but recevied %s.'
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


def synthesis_with_griffinlim(mel_output, _ljspeech_processor):
    # synthesis with griffin-lim
    mel_output = fluid.layers.transpose(
        fluid.layers.squeeze(mel_output, [0]), [1, 0])
    mel_output = np.exp(mel_output.numpy())
    basis = librosa.filters.mel(22050, 1024, 80, fmin=0, fmax=8000)
    inv_basis = np.linalg.pinv(basis)
    spec = np.maximum(1e-10, np.dot(inv_basis, mel_output))

    wav = librosa.core.griffinlim(spec**1.2, hop_length=256, win_length=1024)

    return wav


def synthesis_with_wavenet(mel_output, args):
    with open(args.config_vocoder, 'rt') as f:
        config = yaml.safe_load(f)
    n_mels = config["data"]["n_mels"]
    model_config = config["model"]
    filter_size = model_config["filter_size"]
    upsampling_factors = model_config["upsampling_factors"]
    encoder = UpsampleNet(upsampling_factors)

    n_loop = model_config["n_loop"]
    n_layer = model_config["n_layer"]
    residual_channels = model_config["residual_channels"]
    output_dim = model_config["output_dim"]
    loss_type = model_config["loss_type"]
    log_scale_min = model_config["log_scale_min"]
    decoder = WaveNet(n_loop, n_layer, residual_channels, output_dim, n_mels,
                      filter_size, loss_type, log_scale_min)

    model = ConditionalWavenet(encoder, decoder)

    # load model parameters
    iteration = io.load_parameters(
        model, checkpoint_path=args.checkpoint_vocoder)

    for layer in model.sublayers():
        if isinstance(layer, WeightNormWrapper):
            layer.remove_weight_norm()
    mel_output = fluid.layers.transpose(mel_output, [0, 2, 1])
    wav = model.synthesis(mel_output)
    return wav.numpy()[0]


def synthesis_with_cbhg(mel_output, _ljspeech_processor, cfg):
    with fluid.unique_name.guard():
        model_vocoder = Vocoder(
            cfg['train']['batch_size'], cfg['vocoder']['hidden_size'],
            cfg['audio']['num_mels'], cfg['audio']['n_fft'])
        # Load parameters.
        global_step = io.load_parameters(
            model=model_vocoder, checkpoint_path=args.checkpoint_vocoder)
        model_vocoder.eval()
    mag_pred = model_vocoder(mel_output)
    # synthesis with cbhg
    wav = _ljspeech_processor.inv_spectrogram(
        fluid.layers.transpose(fluid.layers.squeeze(mag_pred, [0]), [1, 0])
        .numpy())
    return wav


def synthesis_with_waveflow(mel_output, args, checkpoint, _ljspeech_processor,
                            place):
    mel_output = fluid.layers.transpose(
        fluid.layers.squeeze(mel_output, [0]), [1, 0])
    mel_output = mel_output.numpy()
    #mel_output = (mel_output - mel_output.min())/(mel_output.max() - mel_output.min())
    #mel_output = 5 * mel_output - 4
    #mel_output = np.log(10) * mel_output

    fluid.enable_dygraph(place)
    args.config = args.config_vocoder
    args.use_fp16 = False
    config = io.add_yaml_config_to_args(args)

    mel_spectrogram = dg.to_variable(mel_output)
    mel_spectrogram = fluid.layers.unsqueeze(mel_spectrogram, [0])

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
    # Print the whole config setting.
    pprint(vars(args))
    synthesis(
        "Life was like a box of chocolates, you never know what you're gonna get.",
        args)
