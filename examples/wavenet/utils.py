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
import numpy as np
import soundfile as sf
import paddle.fluid.dygraph as dg


def make_output_tree(output_dir):
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    state_dir = os.path.join(output_dir, "states")
    if not os.path.exists(state_dir):
        os.makedirs(state_dir)


def valid_model(model, valid_loader, writer, global_step, sample_rate):
    loss = []
    wavs = []
    model.eval()
    for i, batch in enumerate(valid_loader):
        # print("sentence {}".format(i))
        audio_clips, mel_specs, audio_starts = batch
        y_var = model(audio_clips, mel_specs, audio_starts)
        wav_var = model.sample(y_var)
        loss_var = model.loss(y_var, audio_clips)
        loss.append(loss_var.numpy()[0])
        wavs.append(wav_var.numpy()[0])

    average_loss = np.mean(loss)
    writer.add_scalar("valid_loss", average_loss, global_step)
    for i, wav in enumerate(wavs):
        writer.add_audio("valid/sample_{}".format(i), wav, global_step,
                         sample_rate)


def eval_model(model, valid_loader, output_dir, sample_rate):
    model.eval()
    for i, batch in enumerate(valid_loader):
        # print("sentence {}".format(i))
        path = os.path.join(output_dir, "sentence_{}.wav".format(i))
        audio_clips, mel_specs, audio_starts = batch
        wav_var = model.synthesis(mel_specs)
        wav_np = wav_var.numpy()[0]
        sf.write(wav_np, path, samplerate=sample_rate)
        print("generated {}".format(path))


def save_checkpoint(model, optim, checkpoint_dir, global_step):
    checkpoint_path = os.path.join(checkpoint_dir,
                                   "step_{:09d}".format(global_step))
    dg.save_dygraph(model.state_dict(), checkpoint_path)
    dg.save_dygraph(optim.state_dict(), checkpoint_path)
