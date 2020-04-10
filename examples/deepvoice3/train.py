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
import argparse
import ruamel.yaml
import numpy as np
import matplotlib
matplotlib.use("agg")
from matplotlib import cm
import matplotlib.pyplot as plt
import tqdm
import librosa
from librosa import display
import soundfile as sf
from tensorboardX import SummaryWriter

from paddle import fluid
import paddle.fluid.layers as F
import paddle.fluid.dygraph as dg

from parakeet.g2p import en
from parakeet.data import FilterDataset, TransformDataset, FilterDataset
from parakeet.data import DataCargo, PartialyRandomizedSimilarTimeLengthSampler, SequentialSampler
from parakeet.models.deepvoice3 import Encoder, Decoder, Converter, DeepVoice3, ConvSpec
from parakeet.models.deepvoice3.loss import TTSLoss
from parakeet.utils.layer_tools import summary
from parakeet.utils import io

from data import LJSpeechMetaData, DataCollector, Transform
from utils import make_model, eval_model, save_state, make_output_tree, plot_alignment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Deep Voice 3 model with LJSpeech dataset.")
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

    args, _ = parser.parse_known_args()
    with open(args.config, 'rt') as f:
        config = ruamel.yaml.safe_load(f)

    print("Command Line Args: ")
    for k, v in vars(args).items():
        print("{}: {}".format(k, v))

    # =========================dataset=========================
    # construct meta data
    data_root = args.data
    meta = LJSpeechMetaData(data_root)

    # filter it!
    min_text_length = config["meta_data"]["min_text_length"]
    meta = FilterDataset(meta, lambda x: len(x[2]) >= min_text_length)

    # transform meta data into meta data
    transform_config = config["transform"]
    replace_pronounciation_prob = transform_config[
        "replace_pronunciation_prob"]
    sample_rate = transform_config["sample_rate"]
    preemphasis = transform_config["preemphasis"]
    n_fft = transform_config["n_fft"]
    win_length = transform_config["win_length"]
    hop_length = transform_config["hop_length"]
    fmin = transform_config["fmin"]
    fmax = transform_config["fmax"]
    n_mels = transform_config["n_mels"]
    min_level_db = transform_config["min_level_db"]
    ref_level_db = transform_config["ref_level_db"]
    max_norm = transform_config["max_norm"]
    clip_norm = transform_config["clip_norm"]
    transform = Transform(replace_pronounciation_prob, sample_rate,
                          preemphasis, n_fft, win_length, hop_length, fmin,
                          fmax, n_mels, min_level_db, ref_level_db, max_norm,
                          clip_norm)
    ljspeech = TransformDataset(meta, transform)

    # =========================dataiterator=========================
    # use meta data's text length as a sort key for the sampler
    train_config = config["train"]
    batch_size = train_config["batch_size"]
    text_lengths = [len(example[2]) for example in meta]
    sampler = PartialyRandomizedSimilarTimeLengthSampler(text_lengths,
                                                         batch_size)

    # some hyperparameters affect how we process data, so create a data collector!
    model_config = config["model"]
    downsample_factor = model_config["downsample_factor"]
    r = model_config["outputs_per_step"]
    collector = DataCollector(downsample_factor=downsample_factor, r=r)
    ljspeech_loader = DataCargo(
        ljspeech, batch_fn=collector, batch_size=batch_size, sampler=sampler)

    # =========================model=========================
    if args.device == -1:
        place = fluid.CPUPlace()
    else:
        place = fluid.CUDAPlace(args.device)

    with dg.guard(place):
        # =========================model=========================
        n_speakers = model_config["n_speakers"]
        speaker_dim = model_config["speaker_embed_dim"]
        speaker_embed_std = model_config["speaker_embedding_weight_std"]
        n_vocab = en.n_vocab
        embed_dim = model_config["text_embed_dim"]
        linear_dim = 1 + n_fft // 2
        use_decoder_states = model_config[
            "use_decoder_state_for_postnet_input"]
        filter_size = model_config["kernel_size"]
        encoder_channels = model_config["encoder_channels"]
        decoder_channels = model_config["decoder_channels"]
        converter_channels = model_config["converter_channels"]
        dropout = model_config["dropout"]
        padding_idx = model_config["padding_idx"]
        embedding_std = model_config["embedding_weight_std"]
        max_positions = model_config["max_positions"]
        freeze_embedding = model_config["freeze_embedding"]
        trainable_positional_encodings = model_config[
            "trainable_positional_encodings"]
        use_memory_mask = model_config["use_memory_mask"]
        query_position_rate = model_config["query_position_rate"]
        key_position_rate = model_config["key_position_rate"]
        window_backward = model_config["window_backward"]
        window_ahead = model_config["window_ahead"]
        key_projection = model_config["key_projection"]
        value_projection = model_config["value_projection"]
        dv3 = make_model(
            n_speakers, speaker_dim, speaker_embed_std, embed_dim, padding_idx,
            embedding_std, max_positions, n_vocab, freeze_embedding,
            filter_size, encoder_channels, n_mels, decoder_channels, r,
            trainable_positional_encodings, use_memory_mask,
            query_position_rate, key_position_rate, window_backward,
            window_ahead, key_projection, value_projection, downsample_factor,
            linear_dim, use_decoder_states, converter_channels, dropout)
        summary(dv3)

        # =========================loss=========================
        loss_config = config["loss"]
        masked_weight = loss_config["masked_loss_weight"]
        priority_freq = loss_config["priority_freq"]  # Hz
        priority_bin = int(priority_freq / (0.5 * sample_rate) * linear_dim)
        priority_freq_weight = loss_config["priority_freq_weight"]
        binary_divergence_weight = loss_config["binary_divergence_weight"]
        guided_attention_sigma = loss_config["guided_attention_sigma"]
        criterion = TTSLoss(
            masked_weight=masked_weight,
            priority_bin=priority_bin,
            priority_weight=priority_freq_weight,
            binary_divergence_weight=binary_divergence_weight,
            guided_attention_sigma=guided_attention_sigma,
            downsample_factor=downsample_factor,
            r=r)

        # =========================lr_scheduler=========================
        lr_config = config["lr_scheduler"]
        warmup_steps = lr_config["warmup_steps"]
        peak_learning_rate = lr_config["peak_learning_rate"]
        lr_scheduler = dg.NoamDecay(
            1 / (warmup_steps * (peak_learning_rate)**2), warmup_steps)

        # =========================optimizer=========================
        optim_config = config["optimizer"]
        beta1 = optim_config["beta1"]
        beta2 = optim_config["beta2"]
        epsilon = optim_config["epsilon"]
        optim = fluid.optimizer.Adam(
            lr_scheduler,
            beta1,
            beta2,
            epsilon=epsilon,
            parameter_list=dv3.parameters())
        gradient_clipper = fluid.clip.GradientClipByGlobalNorm(0.1)

        # generation
        synthesis_config = config["synthesis"]
        power = synthesis_config["power"]
        n_iter = synthesis_config["n_iter"]

        # =========================link(dataloader, paddle)=========================
        loader = fluid.io.DataLoader.from_generator(
            capacity=10, return_list=True)
        loader.set_batch_generator(ljspeech_loader, places=place)

        # tensorboard & checkpoint preparation
        output_dir = args.output
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        log_dir = os.path.join(output_dir, "log")
        state_dir = os.path.join(output_dir, "states")
        make_output_tree(output_dir)
        writer = SummaryWriter(logdir=log_dir)

        # load parameters and optimizer, and opdate iterations done sofar
        if args.checkpoint is not None:
            iteration = io.load_parameters(
                dv3, optim, checkpoint_path=args.checkpoint)
        else:
            iteration = io.load_parameters(
                dv3, optim, checkpoint_dir=ckpt_dir, iteration=args.iteration)

        # =========================train=========================
        max_iter = train_config["max_iteration"]
        snap_interval = train_config["snap_interval"]
        save_interval = train_config["save_interval"]
        eval_interval = train_config["eval_interval"]

        global_step = iteration + 1
        iterator = iter(tqdm.tqdm(loader))
        while global_step <= max_iter:
            try:
                batch = next(iterator)
            except StopIteration as e:
                iterator = iter(tqdm.tqdm(loader))
                batch = next(iterator)

            dv3.train()
            (text_sequences, text_lengths, text_positions, mel_specs,
             lin_specs, frames, decoder_positions, done_flags) = batch
            downsampled_mel_specs = F.strided_slice(
                mel_specs,
                axes=[1],
                starts=[0],
                ends=[mel_specs.shape[1]],
                strides=[downsample_factor])
            mel_outputs, linear_outputs, alignments, done = dv3(
                text_sequences, text_positions, text_lengths, None,
                downsampled_mel_specs, decoder_positions)

            losses = criterion(mel_outputs, linear_outputs, done, alignments,
                               downsampled_mel_specs, lin_specs, done_flags,
                               text_lengths, frames)
            l = losses["loss"]
            l.backward()
            # record learning rate before updating
            writer.add_scalar("learning_rate",
                              optim._learning_rate.step().numpy(), global_step)
            optim.minimize(l, grad_clip=gradient_clipper)
            optim.clear_gradients()

            # ==================all kinds of tedious things=================
            # record step loss into tensorboard
            step_loss = {k: v.numpy()[0] for k, v in losses.items()}
            tqdm.tqdm.write("global_step: {}\tloss: {}".format(
                global_step, step_loss["loss"]))
            for k, v in step_loss.items():
                writer.add_scalar(k, v, global_step)

            # train state saving, the first sentence in the batch
            if global_step % snap_interval == 0:
                save_state(
                    state_dir,
                    writer,
                    global_step,
                    mel_input=downsampled_mel_specs,
                    mel_output=mel_outputs,
                    lin_input=lin_specs,
                    lin_output=linear_outputs,
                    alignments=alignments,
                    win_length=win_length,
                    hop_length=hop_length,
                    min_level_db=min_level_db,
                    ref_level_db=ref_level_db,
                    power=power,
                    n_iter=n_iter,
                    preemphasis=preemphasis,
                    sample_rate=sample_rate)

            # evaluation
            if global_step % eval_interval == 0:
                sentences = [
                    "Scientists at the CERN laboratory say they have discovered a new particle.",
                    "There's a way to measure the acute emotional intelligence that has never gone out of style.",
                    "President Trump met with other leaders at the Group of 20 conference.",
                    "Generative adversarial network or variational auto-encoder.",
                    "Please call Stella.",
                    "Some have accepted this as a miracle without any physical explanation.",
                ]
                for idx, sent in enumerate(sentences):
                    wav, attn = eval_model(
                        dv3, sent, replace_pronounciation_prob, min_level_db,
                        ref_level_db, power, n_iter, win_length, hop_length,
                        preemphasis)
                    wav_path = os.path.join(
                        state_dir, "waveform",
                        "eval_sample_{:09d}.wav".format(global_step))
                    sf.write(wav_path, wav, sample_rate)
                    writer.add_audio(
                        "eval_sample_{}".format(idx),
                        wav,
                        global_step,
                        sample_rate=sample_rate)
                    attn_path = os.path.join(
                        state_dir, "alignments",
                        "eval_sample_attn_{:09d}.png".format(global_step))
                    plot_alignment(attn, attn_path)
                    writer.add_image(
                        "eval_sample_attn{}".format(idx),
                        cm.viridis(attn),
                        global_step,
                        dataformats="HWC")

            # save checkpoint
            if global_step % save_interval == 0:
                io.save_parameters(ckpt_dir, global_step, dv3, optim)

            global_step += 1
