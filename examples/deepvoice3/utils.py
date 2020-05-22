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

from __future__ import division
import os
import numpy as np
import matplotlib
matplotlib.use("agg")
from matplotlib import cm
import matplotlib.pyplot as plt
import librosa
from scipy import signal
from librosa import display
import soundfile as sf

from paddle import fluid
import paddle.fluid.dygraph as dg
from parakeet.g2p import en


def get_place(device_id):
    """get place from device_id, -1 stands for CPU"""
    if device_id == -1:
        place = fluid.CPUPlace()
    else:
        place = fluid.CUDAPlace(device_id)
    return place


def add_options(parser):
    parser.add_argument("--config", type=str, help="experimrnt config")
    parser.add_argument(
        "--data",
        type=str,
        default="/workspace/datasets/LJSpeech-1.1/",
        help="The path of the LJSpeech dataset.")
    parser.add_argument("--device", type=int, default=-1, help="device to use")

    g = parser.add_mutually_exclusive_group()
    g.add_argument("--checkpoint", type=str, help="checkpoint to resume from.")
    g.add_argument(
        "--iteration",
        type=int,
        help="the iteration of the checkpoint to load from output directory")

    parser.add_argument(
        "output", type=str, default="experiment", help="path to save results")


def make_evaluator(config, text_sequences, output_dir, writer=None):
    c = config["transform"]
    p_replace = c["replace_pronunciation_prob"]
    sample_rate = c["sample_rate"]
    preemphasis = c["preemphasis"]
    win_length = c["win_length"]
    hop_length = c["hop_length"]
    min_level_db = c["min_level_db"]
    ref_level_db = c["ref_level_db"]

    synthesis_config = config["synthesis"]
    power = synthesis_config["power"]
    n_iter = synthesis_config["n_iter"]

    return Evaluator(
        text_sequences,
        p_replace,
        sample_rate,
        preemphasis,
        win_length,
        hop_length,
        min_level_db,
        ref_level_db,
        power,
        n_iter,
        output_dir=output_dir,
        writer=writer)


class Evaluator(object):
    def __init__(self,
                 text_sequences,
                 p_replace,
                 sample_rate,
                 preemphasis,
                 win_length,
                 hop_length,
                 min_level_db,
                 ref_level_db,
                 power,
                 n_iter,
                 output_dir,
                 writer=None):
        self.text_sequences = text_sequences
        self.output_dir = output_dir
        self.writer = writer

        self.p_replace = p_replace
        self.sample_rate = sample_rate
        self.preemphasis = preemphasis
        self.win_length = win_length
        self.hop_length = hop_length
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db

        self.power = power
        self.n_iter = n_iter

    def process_a_sentence(self, model, text):
        text = np.array(
            en.text_to_sequence(
                text, p=self.p_replace), dtype=np.int64)
        length = len(text)
        text_positions = np.arange(1, 1 + length)
        text = np.expand_dims(text, 0)
        text_positions = np.expand_dims(text_positions, 0)

        model.eval()
        if isinstance(model, dg.DataParallel):
            _model = model._layers
        else:
            _model = model
        mel_outputs, linear_outputs, alignments, done = _model.transduce(
            dg.to_variable(text), dg.to_variable(text_positions))

        linear_outputs_np = linear_outputs.numpy()[0].T  # (C, T)

        wav = spec_to_waveform(linear_outputs_np, self.min_level_db,
                               self.ref_level_db, self.power, self.n_iter,
                               self.win_length, self.hop_length,
                               self.preemphasis)
        alignments_np = alignments.numpy()[0]  # batch_size = 1
        return wav, alignments_np

    def __call__(self, model, iteration):
        writer = self.writer
        for i, seq in enumerate(self.text_sequences):
            print("[Eval] synthesizing sentence {}".format(i))
            wav, alignments_np = self.process_a_sentence(model, seq)

            wav_path = os.path.join(
                self.output_dir,
                "eval_sample_{}_step_{:09d}.wav".format(i, iteration))
            sf.write(wav_path, wav, self.sample_rate)
            if writer is not None:
                writer.add_audio(
                    "eval_sample_{}".format(i),
                    wav,
                    iteration,
                    sample_rate=self.sample_rate)
            attn_path = os.path.join(
                self.output_dir,
                "eval_sample_{}_step_{:09d}.png".format(i, iteration))
            plot_alignment(alignments_np, attn_path)
            if writer is not None:
                writer.add_image(
                    "eval_sample_attn_{}".format(i),
                    cm.viridis(alignments_np),
                    iteration,
                    dataformats="HWC")


def make_state_saver(config, output_dir, writer=None):
    c = config["transform"]
    p_replace = c["replace_pronunciation_prob"]
    sample_rate = c["sample_rate"]
    preemphasis = c["preemphasis"]
    win_length = c["win_length"]
    hop_length = c["hop_length"]
    min_level_db = c["min_level_db"]
    ref_level_db = c["ref_level_db"]

    synthesis_config = config["synthesis"]
    power = synthesis_config["power"]
    n_iter = synthesis_config["n_iter"]

    return StateSaver(p_replace, sample_rate, preemphasis, win_length,
                      hop_length, min_level_db, ref_level_db, power, n_iter,
                      output_dir, writer)


class StateSaver(object):
    def __init__(self,
                 p_replace,
                 sample_rate,
                 preemphasis,
                 win_length,
                 hop_length,
                 min_level_db,
                 ref_level_db,
                 power,
                 n_iter,
                 output_dir,
                 writer=None):
        self.output_dir = output_dir
        self.writer = writer

        self.p_replace = p_replace
        self.sample_rate = sample_rate
        self.preemphasis = preemphasis
        self.win_length = win_length
        self.hop_length = hop_length
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db

        self.power = power
        self.n_iter = n_iter

    def __call__(self, outputs, inputs, iteration):
        mel_output, lin_output, alignments, done_output = outputs
        mel_input, lin_input = inputs
        writer = self.writer

        # mel spectrogram
        mel_input = mel_input[0].numpy().T
        mel_output = mel_output[0].numpy().T

        path = os.path.join(self.output_dir, "mel_spec")
        plt.figure(figsize=(10, 3))
        display.specshow(mel_input)
        plt.colorbar()
        plt.title("mel_input")
        plt.savefig(
            os.path.join(path, "target_mel_spec_step_{:09d}.png".format(
                iteration)))
        plt.close()

        if writer is not None:
            writer.add_image(
                "target/mel_spec",
                cm.viridis(mel_input),
                iteration,
                dataformats="HWC")

        plt.figure(figsize=(10, 3))
        display.specshow(mel_output)
        plt.colorbar()
        plt.title("mel_output")
        plt.savefig(
            os.path.join(path, "predicted_mel_spec_step_{:09d}.png".format(
                iteration)))
        plt.close()

        if writer is not None:
            writer.add_image(
                "predicted/mel_spec",
                cm.viridis(mel_output),
                iteration,
                dataformats="HWC")

        # linear spectrogram
        lin_input = lin_input[0].numpy().T
        lin_output = lin_output[0].numpy().T
        path = os.path.join(self.output_dir, "lin_spec")

        plt.figure(figsize=(10, 3))
        display.specshow(lin_input)
        plt.colorbar()
        plt.title("mel_input")
        plt.savefig(
            os.path.join(path, "target_lin_spec_step_{:09d}.png".format(
                iteration)))
        plt.close()

        if writer is not None:
            writer.add_image(
                "target/lin_spec",
                cm.viridis(lin_input),
                iteration,
                dataformats="HWC")

        plt.figure(figsize=(10, 3))
        display.specshow(lin_output)
        plt.colorbar()
        plt.title("mel_input")
        plt.savefig(
            os.path.join(path, "predicted_lin_spec_step_{:09d}.png".format(
                iteration)))
        plt.close()

        if writer is not None:
            writer.add_image(
                "predicted/lin_spec",
                cm.viridis(lin_output),
                iteration,
                dataformats="HWC")

        # alignment
        path = os.path.join(self.output_dir, "alignments")
        alignments = alignments[:, 0, :, :].numpy()
        for idx, attn_layer in enumerate(alignments):
            save_path = os.path.join(
                path, "train_attn_layer_{}_step_{}.png".format(idx, iteration))
            plot_alignment(attn_layer, save_path)

            if writer is not None:
                writer.add_image(
                    "train_attn/layer_{}".format(idx),
                    cm.viridis(attn_layer),
                    iteration,
                    dataformats="HWC")

        # synthesize waveform
        wav = spec_to_waveform(
            lin_output, self.min_level_db, self.ref_level_db, self.power,
            self.n_iter, self.win_length, self.hop_length, self.preemphasis)
        path = os.path.join(self.output_dir, "waveform")
        save_path = os.path.join(
            path, "train_sample_step_{:09d}.wav".format(iteration))
        sf.write(save_path, wav, self.sample_rate)

        if writer is not None:
            writer.add_audio(
                "train_sample", wav, iteration, sample_rate=self.sample_rate)


def spec_to_waveform(spec, min_level_db, ref_level_db, power, n_iter,
                     win_length, hop_length, preemphasis):
    """Convert output linear spec to waveform using griffin-lim vocoder.
    
    Args:
        spec (ndarray): the output linear spectrogram, shape(C, T), where C means n_fft, T means frames.
    """
    denoramlized = np.clip(spec, 0, 1) * (-min_level_db) + min_level_db
    lin_scaled = np.exp((denoramlized + ref_level_db) / 20 * np.log(10))
    wav = librosa.griffinlim(
        lin_scaled**power,
        n_iter=n_iter,
        hop_length=hop_length,
        win_length=win_length)
    if preemphasis > 0:
        wav = signal.lfilter([1.], [1., -preemphasis], wav)
    wav = np.clip(wav, -1.0, 1.0)
    return wav


def make_output_tree(output_dir):
    print("creating output tree: {}".format(output_dir))
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    state_dir = os.path.join(output_dir, "states")
    eval_dir = os.path.join(output_dir, "eval")

    for x in [ckpt_dir, state_dir, eval_dir]:
        if not os.path.exists(x):
            os.makedirs(x)
    for x in ["alignments", "waveform", "lin_spec", "mel_spec"]:
        p = os.path.join(state_dir, x)
        if not os.path.exists(p):
            os.makedirs(p)


def plot_alignment(alignment, path):
    """
    Plot an attention layer's alignment for a sentence.
    alignment: shape(T_dec, T_enc).
    """

    plt.figure()
    plt.imshow(alignment)
    plt.colorbar()
    plt.xlabel('Encoder timestep')
    plt.ylabel('Decoder timestep')
    plt.savefig(path)
    plt.close()
