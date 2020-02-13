import os
import argparse
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import tqdm
import librosa
from scipy import signal
from librosa import display
import soundfile as sf
from tensorboardX import SummaryWriter

from paddle import fluid
import paddle.fluid.layers as F
import paddle.fluid.dygraph as dg

from parakeet.g2p import en
from parakeet.models.Rdeepvoice3.encoder import ConvSpec
from parakeet.data import FilterDataset, TransformDataset, FilterDataset, DatasetMixin
from parakeet.data import DataCargo, PartialyRandomizedSimilarTimeLengthSampler, SequentialSampler
from parakeet.models.Rdeepvoice3 import Encoder, Decoder, Converter, DeepVoice3
from parakeet.models.Rdeepvoice3.loss import TTSLoss
from parakeet.modules.weight_norm_wrapper import WeightNormWrapper
from parakeet.utils.layer_tools import summary

from data_validate import LJSpeechMetaData, DataCollector, Transform
from utils import make_model, eval_model, plot_alignment, plot_alignments, save_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a deepvoice 3 model with LJSpeech")
    parser.add_argument("-o",
                        "--output",
                        type=str,
                        default="result",
                        help="The directory to save result.")
    parser.add_argument("-d",
                        "--data",
                        type=str,
                        default="/workspace/datasets/ljs_dv3",
                        help="The path of the LJSpeech dataset.")
    parser.add_argument("-r", "--resume", type=str, help="checkpoint to load")
    args, _ = parser.parse_known_args()

    # =========================dataset=========================
    data_root = args.data

    meta = LJSpeechMetaData(data_root)  # construct meta data
    #meta = FilterDataset(meta, lambda x: len(x[3]) >= 20)  # filter it!

    transform = Transform()
    ljspeech = TransformDataset(meta, transform)

    # =========================dataiterator=========================
    # use meta data's text length as a sort key
    # which is used in sampler
    text_lengths = [len(example[3]) for example in meta]
    # some hyperparameters affect how we process data, so create a data collector!
    collector = DataCollector(downsample_factor=4., r=1)
    ljspeech_loader = DataCargo(ljspeech,
                                batch_fn=collector,
                                batch_size=16,
                                sampler=SequentialSampler(ljspeech))
    # sampler=PartialyRandomizedSimilarTimeLengthSampler(text_lengths,
    #                                                    batch_size=32))

    # ljspeech_iterator = ljspeech_loader() # if you want to inspect it!
    # for i in range(3):
    #     batch = next(ljspeech_iterator)
    #     print(batch)

    # =========================model=========================
    sample_rate = 22050

    n_speakers = 1
    speaker_dim = 16
    n_vocab = en.n_vocab
    embed_dim = 256
    mel_dim = 80

    downsample_factor = 4
    r = 1

    linear_dim = 1 + 1024 // 2
    use_decoder_states = True
    filter_size = 3

    encoder_channels = 512
    decoder_channels = 256
    converter_channels = 256

    dropout = 0.  #0.050000000000000044

    place = fluid.CPUPlace()
    with dg.guard(place):
        # =========================model=========================
        dv3 = make_model(n_speakers, speaker_dim, n_vocab, embed_dim, mel_dim,
                         downsample_factor, r, linear_dim, use_decoder_states,
                         filter_size, encoder_channels, decoder_channels,
                         converter_channels, dropout)

        # =========================loss=========================
        priority_freq = 3000  # Hz
        priority_bin = int(priority_freq / (0.5 * sample_rate) * linear_dim)
        criterion = TTSLoss(masked_weight=.5,
                            priority_bin=priority_bin,
                            priority_weight=.0,
                            binary_divergence_weight=.1,
                            guided_attention_sigma=.2,
                            downsample_factor=downsample_factor,
                            r=r)
        # summary(dv3)

        # =========================lr_scheduler=========================
        warmup_steps = 4000
        peak_learning_rate = 5e-4
        lr_scheduler = dg.NoamDecay(d_model=1 / (warmup_steps *
                                                 (peak_learning_rate)**2),
                                    warmup_steps=warmup_steps)

        # =========================optimizer=========================
        beta1, beta2 = 0.5, 0.9
        epsilon = 1e-6
        optim = fluid.optimizer.Adam(lr_scheduler,
                                     beta1,
                                     beta2,
                                     epsilon=1e-6,
                                     parameter_list=dv3.parameters())

        # =========================link(dataloader, paddle)=========================
        # CAUTION: it does not return a DataLoader
        loader = fluid.io.DataLoader.from_generator(capacity=10,
                                                    return_list=True)
        loader.set_batch_generator(ljspeech_loader, places=place)

        # tensorboard & checkpoint preparation
        output_dir = args.output
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
        writer = SummaryWriter(logdir=log_dir)

        # DEBUG
        resume_path = args.resume
        if resume_path is not None:
            state, _ = dg.load_dygraph(args.resume)
            dv3.set_dict(state)

        # =========================train=========================
        epoch = 3000
        global_step = 1

        average_loss = {"mel": 0, "lin": 0, "done": 0, "attn": 0}
        epoch_loss = {"mel": 0, "lin": 0, "done": 0, "attn": 0}
        for j in range(epoch):
            for i, batch in tqdm.tqdm(enumerate(loader)):
                dv3.train()  # switch to train
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

                # print("========")
                # print("text lengths: {}".format(text_lengths.numpy()))
                # print("n frames: {}".format(frames.numpy()))
                # print("[mel] mel's shape: {}; "
                #       "downsampled mel's shape: {}; "
                #       "output's shape: {}".format(mel_specs.shape,
                #                                   downsampled_mel_specs.shape,
                #                                   mel_outputs.shape))
                # print("[lin] lin's shape: {}; "
                #       "output's shape{}".format(lin_specs.shape,
                #                                 linear_outputs.shape))
                # print("[attn]: alignments's shape: {}".format(alignments.shape))
                # print("[done]: input done flag's shape: {}; "
                #       "output done flag's shape: {}".format(
                #           done_flags.shape, done.shape))

                losses = criterion(mel_outputs, linear_outputs, done,
                                   alignments, downsampled_mel_specs,
                                   lin_specs, done_flags, text_lengths, frames)
                for k in epoch_loss.keys():
                    epoch_loss[k] += losses[k].numpy()[0]
                    average_loss[k] += losses[k].numpy()[0]
                global_step += 1

                # train state saving, the first sentence in the batch
                if global_step > 0 and global_step % 10 == 0:
                    linear_outputs_np = linear_outputs.numpy()[0].T
                    denoramlized = np.clip(linear_outputs_np, 0,
                                           1) * 100. - 100.
                    lin_scaled = np.exp((denoramlized + 20) / 20 * np.log(10))
                    wav = librosa.griffinlim(lin_scaled**1.4,
                                             n_iter=32,
                                             hop_length=256,
                                             win_length=1024)

                    save_state(state_dir,
                               global_step,
                               mel_input=mel_specs.numpy()[0].T,
                               mel_output=mel_outputs.numpy()[0].T,
                               lin_input=lin_specs.numpy()[0].T,
                               lin_output=linear_outputs.numpy()[0].T,
                               alignments=alignments.numpy()[:, 0, :, :],
                               wav=wav)

                # evaluation
                if global_step > 0 and global_step % 10 == 0:
                    wav, attn = eval_model(
                        dv3,
                        "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"
                    )
                    wav_path = os.path.join(
                        state_dir, "waveform",
                        "eval_sample_{}.wav".format(global_step))
                    sf.write(wav_path, wav, 22050)
                    attn_path = os.path.join(
                        state_dir, "alignments",
                        "eval_sample_attn_{}.png".format(global_step))
                    plot_alignment(attn, attn_path)

                # for tensorboard writer, if you want more, write more
                # cause you are in the process
                step_loss = {k: v.numpy()[0] for k, v in losses.items()}
                for k, v in step_loss.items():
                    writer.add_scalar(k, v, global_step)

                # save checkpoint
                if global_step % 1000 == 0:
                    for i, attn_layer in enumerate(
                            alignments.numpy()[:, 0, :, :]):
                        plt.figure()
                        plt.imshow(attn_layer)
                        plt.xlabel("encoder_timesteps")
                        plt.ylabel("decoder_timesteps")
                        plt.savefig("results3/step_{}_layer_{}.png".format(
                            global_step, i),
                                    format="png")
                        plt.close()

                # print(step_loss)

                if global_step % 100 == 0:
                    for k in epoch_loss.keys():
                        average_loss[k] /= 100
                    print("[average_loss] ",
                          "global_step: {}".format(global_step), average_loss)
                    average_loss = {"mel": 0, "lin": 0, "done": 0, "attn": 0}

                l = criterion.compose_loss(losses)
                l.backward()
                # print("loss: ", l.numpy()[0])
                optim.minimize(
                    l,
                    grad_clip=fluid.dygraph_grad_clip.GradClipByGlobalNorm(
                        0.1))
                dv3.clear_gradients()

                if global_step % 10000 == 0:
                    dg.save_dygraph(dv3.state_dict(),
                                    os.path.join(ckpt_dir, "dv3"))
                    dg.save_dygraph(optim.state_dict(),
                                    os.path.join(ckpt_dir, "dv3"))

            for k in epoch_loss.keys():
                epoch_loss[k] /= (i + 1)
            print("[epoch_loss] ", "epoch: {}".format(j + 1), epoch_loss)
            epoch_loss = {"mel": 0, "lin": 0, "done": 0, "attn": 0}
