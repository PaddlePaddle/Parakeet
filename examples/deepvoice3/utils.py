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
from matplotlib import cm
import matplotlib.pyplot as plt
import librosa
from scipy import signal
from librosa import display
import soundfile as sf

from paddle import fluid
import paddle.fluid.dygraph as dg
import paddle.fluid.initializer as I

from parakeet.g2p import en
from parakeet.models.deepvoice3.encoder import ConvSpec
from parakeet.models.deepvoice3 import Encoder, Decoder, Converter, DeepVoice3, WindowRange
from parakeet.utils.layer_tools import freeze


@fluid.framework.dygraph_only
def make_model(n_speakers, speaker_dim, speaker_embed_std, embed_dim,
               padding_idx, embedding_std, max_positions, n_vocab,
               freeze_embedding, filter_size, encoder_channels, mel_dim,
               decoder_channels, r, trainable_positional_encodings,
               use_memory_mask, query_position_rate, key_position_rate,
               window_behind, window_ahead, key_projection, value_projection,
               downsample_factor, linear_dim, use_decoder_states,
               converter_channels, dropout):
    """just a simple function to create a deepvoice 3 model"""
    if n_speakers > 1:
        spe = dg.Embedding(
            (n_speakers, speaker_dim),
            param_attr=I.Normal(scale=speaker_embed_std))
    else:
        spe = None

    h = encoder_channels
    k = filter_size
    encoder_convolutions = (
        ConvSpec(h, k, 1),
        ConvSpec(h, k, 3),
        ConvSpec(h, k, 9),
        ConvSpec(h, k, 27),
        ConvSpec(h, k, 1),
        ConvSpec(h, k, 3),
        ConvSpec(h, k, 9),
        ConvSpec(h, k, 27),
        ConvSpec(h, k, 1),
        ConvSpec(h, k, 3), )
    enc = Encoder(
        n_vocab,
        embed_dim,
        n_speakers,
        speaker_dim,
        padding_idx=None,
        embedding_weight_std=embedding_std,
        convolutions=encoder_convolutions,
        dropout=dropout)
    if freeze_embedding:
        freeze(enc.embed)

    h = decoder_channels
    prenet_convolutions = (ConvSpec(h, k, 1), ConvSpec(h, k, 3))
    attentive_convolutions = (
        ConvSpec(h, k, 1),
        ConvSpec(h, k, 3),
        ConvSpec(h, k, 9),
        ConvSpec(h, k, 27),
        ConvSpec(h, k, 1), )
    attention = [True, False, False, False, True]
    force_monotonic_attention = [True, False, False, False, True]
    dec = Decoder(
        n_speakers,
        speaker_dim,
        embed_dim,
        mel_dim,
        r=r,
        max_positions=max_positions,
        preattention=prenet_convolutions,
        convolutions=attentive_convolutions,
        attention=attention,
        dropout=dropout,
        use_memory_mask=use_memory_mask,
        force_monotonic_attention=force_monotonic_attention,
        query_position_rate=query_position_rate,
        key_position_rate=key_position_rate,
        window_range=WindowRange(window_behind, window_ahead),
        key_projection=key_projection,
        value_projection=value_projection)
    if not trainable_positional_encodings:
        freeze(dec.embed_keys_positions)
        freeze(dec.embed_query_positions)

    h = converter_channels
    postnet_convolutions = (
        ConvSpec(h, k, 1),
        ConvSpec(h, k, 3),
        ConvSpec(2 * h, k, 1),
        ConvSpec(2 * h, k, 3), )
    cvt = Converter(
        n_speakers,
        speaker_dim,
        dec.state_dim if use_decoder_states else mel_dim,
        linear_dim,
        time_upsampling=downsample_factor,
        convolutions=postnet_convolutions,
        dropout=dropout)
    dv3 = DeepVoice3(enc, dec, cvt, spe, use_decoder_states)
    return dv3


@fluid.framework.dygraph_only
def eval_model(model, text, replace_pronounciation_prob, min_level_db,
               ref_level_db, power, n_iter, win_length, hop_length,
               preemphasis):
    """generate waveform from text using a deepvoice 3 model"""
    text = np.array(
        en.text_to_sequence(
            text, p=replace_pronounciation_prob),
        dtype=np.int64)
    length = len(text)
    print("text sequence's length: {}".format(length))
    text_positions = np.arange(1, 1 + length)

    text = np.expand_dims(text, 0)
    text_positions = np.expand_dims(text_positions, 0)
    model.eval()
    mel_outputs, linear_outputs, alignments, done = model.transduce(
        dg.to_variable(text), dg.to_variable(text_positions))

    linear_outputs_np = linear_outputs.numpy()[0].T  # (C, T)
    wav = spec_to_waveform(linear_outputs_np, min_level_db, ref_level_db,
                           power, n_iter, win_length, hop_length, preemphasis)
    alignments_np = alignments.numpy()[0]  # batch_size = 1
    print("linear_outputs's shape: ", linear_outputs_np.shape)
    print("alignmnets' shape:", alignments.shape)
    return wav, alignments_np


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
    return wav


def make_output_tree(output_dir):
    print("creating output tree: {}".format(output_dir))
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    state_dir = os.path.join(output_dir, "states")
    log_dir = os.path.join(output_dir, "log")

    for x in [ckpt_dir, state_dir]:
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


def save_state(save_dir,
               writer,
               global_step,
               mel_input=None,
               mel_output=None,
               lin_input=None,
               lin_output=None,
               alignments=None,
               win_length=1024,
               hop_length=256,
               min_level_db=-100,
               ref_level_db=20,
               power=1.4,
               n_iter=32,
               preemphasis=0.97,
               sample_rate=22050):
    """Save training intermediate results. Save states for the first sentence in the batch, including
    mel_spec(predicted, target), lin_spec(predicted, target), attn, waveform.
    
    Args:
        save_dir (str): directory to save results.
        writer (SummaryWriter): tensorboardX summary writer
        global_step (int): global step.
        mel_input (Variable, optional): Defaults to None. Shape(B, T_mel, C_mel)
        mel_output (Variable, optional): Defaults to None. Shape(B, T_mel, C_mel)
        lin_input (Variable, optional): Defaults to None. Shape(B, T_lin, C_lin)
        lin_output (Variable, optional): Defaults to None. Shape(B, T_lin, C_lin)
        alignments (Variable, optional): Defaults to None. Shape(N, B, T_dec, C_enc)
        wav ([type], optional): Defaults to None. [description]
    """

    if mel_input is not None and mel_output is not None:
        mel_input = mel_input[0].numpy().T
        mel_output = mel_output[0].numpy().T

        path = os.path.join(save_dir, "mel_spec")
        plt.figure(figsize=(10, 3))
        display.specshow(mel_input)
        plt.colorbar()
        plt.title("mel_input")
        plt.savefig(
            os.path.join(path, "target_mel_spec_step{:09d}.png".format(
                global_step)))
        plt.close()

        writer.add_image(
            "target/mel_spec",
            cm.viridis(mel_input),
            global_step,
            dataformats="HWC")

        plt.figure(figsize=(10, 3))
        display.specshow(mel_output)
        plt.colorbar()
        plt.title("mel_output")
        plt.savefig(
            os.path.join(path, "predicted_mel_spec_step{:09d}.png".format(
                global_step)))
        plt.close()

        writer.add_image(
            "predicted/mel_spec",
            cm.viridis(mel_output),
            global_step,
            dataformats="HWC")

    if lin_input is not None and lin_output is not None:
        lin_input = lin_input[0].numpy().T
        lin_output = lin_output[0].numpy().T
        path = os.path.join(save_dir, "lin_spec")

        plt.figure(figsize=(10, 3))
        display.specshow(lin_input)
        plt.colorbar()
        plt.title("mel_input")
        plt.savefig(
            os.path.join(path, "target_lin_spec_step{:09d}.png".format(
                global_step)))
        plt.close()

        writer.add_image(
            "target/lin_spec",
            cm.viridis(lin_input),
            global_step,
            dataformats="HWC")

        plt.figure(figsize=(10, 3))
        display.specshow(lin_output)
        plt.colorbar()
        plt.title("mel_input")
        plt.savefig(
            os.path.join(path, "predicted_lin_spec_step{:09d}.png".format(
                global_step)))
        plt.close()

        writer.add_image(
            "predicted/lin_spec",
            cm.viridis(lin_output),
            global_step,
            dataformats="HWC")

    if alignments is not None and len(alignments.shape) == 4:
        path = os.path.join(save_dir, "alignments")
        alignments = alignments[:, 0, :, :].numpy()
        for idx, attn_layer in enumerate(alignments):
            save_path = os.path.join(
                path,
                "train_attn_layer_{}_step_{}.png".format(idx, global_step))
            plot_alignment(attn_layer, save_path)

            writer.add_image(
                "train_attn/layer_{}".format(idx),
                cm.viridis(attn_layer),
                global_step,
                dataformats="HWC")

    if lin_output is not None:
        wav = spec_to_waveform(lin_output, min_level_db, ref_level_db, power,
                               n_iter, win_length, hop_length, preemphasis)
        path = os.path.join(save_dir, "waveform")
        save_path = os.path.join(
            path, "train_sample_step_{:09d}.wav".format(global_step))
        sf.write(save_path, wav, sample_rate)
        writer.add_audio(
            "train_sample", wav, global_step, sample_rate=sample_rate)
