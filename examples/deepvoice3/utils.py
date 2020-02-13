import os
import numpy as np
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
        spe = dg.Embedding((n_speakers, speaker_dim),
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
        ConvSpec(h, k, 3),
    )
    enc = Encoder(n_vocab,
                  embed_dim,
                  n_speakers,
                  speaker_dim,
                  padding_idx=padding_idx,
                  embedding_weight_std=embedding_std,
                  convolutions=encoder_convolutions,
                  max_positions=max_positions,
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
        ConvSpec(h, k, 1),
    )
    attention = [True, False, False, False, True]
    force_monotonic_attention = [True, False, False, False, True]
    dec = Decoder(n_speakers,
                  speaker_dim,
                  embed_dim,
                  mel_dim,
                  r=r,
                  max_positions=max_positions,
                  padding_idx=padding_idx,
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
        ConvSpec(2 * h, k, 3),
    )
    cvt = Converter(n_speakers,
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
    text = np.array(en.text_to_sequence(text, p=replace_pronounciation_prob),
                    dtype=np.int64)
    length = len(text)
    print("text sequence's length: {}".format(length))
    text_positions = np.arange(1, 1 + length)

    text = np.expand_dims(text, 0)
    text_positions = np.expand_dims(text_positions, 0)
    mel_outputs, linear_outputs, alignments, done = model.transduce(
        dg.to_variable(text), dg.to_variable(text_positions))
    linear_outputs_np = linear_outputs.numpy()[0].T  # (C, T)
    print("linear_outputs's shape: ", linear_outputs_np.shape)

    denoramlized = np.clip(linear_outputs_np, 0,
                           1) * (-min_level_db) + min_level_db
    lin_scaled = np.exp((denoramlized + ref_level_db) / 20 * np.log(10))
    wav = librosa.griffinlim(lin_scaled**power,
                             n_iter=n_iter,
                             hop_length=hop_length,
                             win_length=win_length)
    wav = signal.lfilter([1.], [1., -preemphasis], wav)

    print("alignmnets' shape:", alignments.shape)
    alignments_np = alignments.numpy()[0].T
    return wav, alignments_np


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


def plot_alignment(alignment, path, info=None):
    """
    Plot an attention layer's alignment for a sentence.
    alignment: shape(T_enc, T_dec), and T_enc is flipped
    """

    fig, ax = plt.subplots()
    im = ax.imshow(alignment,
                   aspect='auto',
                   origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_alignments(alignments, save_dir, global_step):
    """
    Plot alignments for a sentence when training, we just pick the first 
    sentence. Each layer is plot separately. 
    alignments: shape(N, T_dec, T_enc)
    """
    n_layers = alignments.shape[0]
    for i, alignment in enumerate(alignments):
        alignment = alignment.T

        path = os.path.join(save_dir, "layer_{}".format(i))
        if not os.path.exists(path):
            os.makedirs(path)
        fname = os.path.join(path, "step_{:09d}".format(global_step))
        plot_alignment(alignment, fname)

    average_alignment = np.mean(alignments, axis=0).T
    path = os.path.join(save_dir, "average")
    if not os.path.exists(path):
        os.makedirs(path)
    fname = os.path.join(path, "step_{:09d}.png".format(global_step))
    plot_alignment(average_alignment, fname)


def save_state(save_dir,
               global_step,
               mel_input=None,
               mel_output=None,
               lin_input=None,
               lin_output=None,
               alignments=None,
               wav=None):

    if mel_input is not None and mel_output is not None:
        path = os.path.join(save_dir, "mel_spec")
        if not os.path.exists(path):
            os.makedirs(path)

        plt.figure(figsize=(10, 3))
        display.specshow(mel_input)
        plt.colorbar()
        plt.title("mel_input")
        plt.savefig(
            os.path.join(path,
                         "target_mel_spec_step{:09d}".format(global_step)))
        plt.close()

        plt.figure(figsize=(10, 3))
        display.specshow(mel_output)
        plt.colorbar()
        plt.title("mel_input")
        plt.savefig(
            os.path.join(path,
                         "predicted_mel_spec_step{:09d}".format(global_step)))
        plt.close()

    if lin_input is not None and lin_output is not None:
        path = os.path.join(save_dir, "lin_spec")
        if not os.path.exists(path):
            os.makedirs(path)

        plt.figure(figsize=(10, 3))
        display.specshow(lin_input)
        plt.colorbar()
        plt.title("mel_input")
        plt.savefig(
            os.path.join(path,
                         "target_lin_spec_step{:09d}".format(global_step)))
        plt.close()

        plt.figure(figsize=(10, 3))
        display.specshow(lin_output)
        plt.colorbar()
        plt.title("mel_input")
        plt.savefig(
            os.path.join(path,
                         "predicted_lin_spec_step{:09d}".format(global_step)))
        plt.close()

    if alignments is not None and len(alignments.shape) == 3:
        path = os.path.join(save_dir, "alignments")
        if not os.path.exists(path):
            os.makedirs(path)
        plot_alignments(alignments, path, global_step)

    if wav is not None:
        path = os.path.join(save_dir, "waveform")
        if not os.path.exists(path):
            os.makedirs(path)
        sf.write(
            os.path.join(path, "sample_step_{:09d}.wav".format(global_step)),
            wav, 22050)
