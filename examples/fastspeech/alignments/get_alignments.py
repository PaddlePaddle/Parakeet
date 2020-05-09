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
from parakeet.g2p.en import text_to_sequence
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
from ruamel import yaml
import pickle
from pathlib import Path
import argparse
from pprint import pprint
from collections import OrderedDict
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
from parakeet.models.transformer_tts.utils import *
from parakeet import audio
from parakeet.models.transformer_tts import TransformerTTS
from parakeet.models.fastspeech.utils import get_alignment
from parakeet.utils import io


def add_config_options_to_parser(parser):
    parser.add_argument("--config", type=str, help="path of the config file")
    parser.add_argument("--use_gpu", type=int, default=0, help="device to use")
    parser.add_argument("--data", type=str, help="path of LJspeech dataset")

    parser.add_argument(
        "--checkpoint_transformer",
        type=str,
        help="transformer_tts checkpoint to synthesis")

    parser.add_argument(
        "--output",
        type=str,
        default="./alignments",
        help="path to save experiment results")


def alignments(args):
    local_rank = dg.parallel.Env().local_rank
    place = (fluid.CUDAPlace(local_rank) if args.use_gpu else fluid.CPUPlace())

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    with dg.guard(place):
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

        # get text data
        root = Path(args.data)
        csv_path = root.joinpath("metadata.csv")
        table = pd.read_csv(
            csv_path,
            sep="|",
            header=None,
            quoting=csv.QUOTE_NONE,
            names=["fname", "raw_text", "normalized_text"])
        ljspeech_processor = audio.AudioProcessor(
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

        pbar = tqdm(range(len(table)))
        alignments = OrderedDict()
        for i in pbar:
            fname, raw_text, normalized_text = table.iloc[i]
            # init input
            text = np.asarray(text_to_sequence(normalized_text))
            text = fluid.layers.unsqueeze(dg.to_variable(text), [0])
            pos_text = np.arange(1, text.shape[1] + 1)
            pos_text = fluid.layers.unsqueeze(dg.to_variable(pos_text), [0])
            wav = ljspeech_processor.load_wav(
                os.path.join(args.data, 'wavs', fname + ".wav"))
            mel_input = ljspeech_processor.melspectrogram(wav).astype(
                np.float32)
            mel_input = np.transpose(mel_input, axes=(1, 0))
            mel_input = fluid.layers.unsqueeze(dg.to_variable(mel_input), [0])
            mel_lens = mel_input.shape[1]

            dec_slf_mask = get_triu_tensor(mel_input,
                                           mel_input).astype(np.float32)
            dec_slf_mask = np.expand_dims(dec_slf_mask, axis=0)
            dec_slf_mask = fluid.layers.cast(
                dg.to_variable(dec_slf_mask != 0), np.float32) * (-2**32 + 1)
            pos_mel = np.arange(1, mel_input.shape[1] + 1)
            pos_mel = fluid.layers.unsqueeze(dg.to_variable(pos_mel), [0])
            mel_pred, postnet_pred, attn_probs, stop_preds, attn_enc, attn_dec = model(
                text, mel_input, pos_text, pos_mel, dec_slf_mask)
            mel_input = fluid.layers.concat(
                [mel_input, postnet_pred[:, -1:, :]], axis=1)

            alignment, _ = get_alignment(attn_probs, mel_lens,
                                         network_cfg['decoder_num_head'])
            alignments[fname] = alignment
        with open(args.output + '.txt', "wb") as f:
            pickle.dump(alignments, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Get alignments from TransformerTTS model")
    add_config_options_to_parser(parser)
    args = parser.parse_args()
    alignments(args)
