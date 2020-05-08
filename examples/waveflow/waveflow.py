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

import itertools
import os
import time

import numpy as np
import paddle.fluid.dygraph as dg
from paddle import fluid
from scipy.io.wavfile import write

from parakeet.utils import io
from parakeet.modules import weight_norm
from parakeet.models.waveflow import WaveFlowLoss, WaveFlowModule
from data import LJSpeech
import utils


class WaveFlow():
    """Wrapper class of WaveFlow model that supports multiple APIs.

    This module provides APIs for model building, training, validation,
    inference, benchmarking, and saving.

    Args:
        config (obj): config info.
        checkpoint_dir (str): path for checkpointing.
        parallel (bool, optional): whether use multiple GPUs for training.
            Defaults to False.
        rank (int, optional): the rank of the process in a multi-process
            scenario. Defaults to 0.
        nranks (int, optional): the total number of processes. Defaults to 1.
        tb_logger (obj, optional): logger to visualize metrics.
            Defaults to None.

    Returns:
        WaveFlow
    """

    def __init__(self,
                 config,
                 checkpoint_dir,
                 parallel=False,
                 rank=0,
                 nranks=1,
                 tb_logger=None):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.parallel = parallel
        self.rank = rank
        self.nranks = nranks
        self.tb_logger = tb_logger
        self.dtype = "float16" if config.use_fp16 else "float32"

    def build(self, training=True):
        """Initialize the model.

        Args:
            training (bool, optional): Whether the model is built for training or inference.
                Defaults to True.

        Returns:
            None
        """
        config = self.config
        dataset = LJSpeech(config, self.nranks, self.rank)
        self.trainloader = dataset.trainloader
        self.validloader = dataset.validloader

        waveflow = WaveFlowModule(config)

        if training:
            optimizer = fluid.optimizer.AdamOptimizer(
                learning_rate=config.learning_rate,
                parameter_list=waveflow.parameters())

            # Load parameters.
            iteration = io.load_parameters(
                model=waveflow,
                optimizer=optimizer,
                checkpoint_dir=self.checkpoint_dir,
                iteration=config.iteration,
                checkpoint_path=config.checkpoint)
            print("Rank {}: checkpoint loaded.".format(self.rank))

            # Data parallelism.
            if self.parallel:
                strategy = dg.parallel.prepare_context()
                waveflow = dg.parallel.DataParallel(waveflow, strategy)

            self.waveflow = waveflow
            self.optimizer = optimizer
            self.criterion = WaveFlowLoss(config.sigma)

        else:
            # Load parameters.
            iteration = io.load_parameters(
                model=waveflow,
                checkpoint_dir=self.checkpoint_dir,
                iteration=config.iteration,
                checkpoint_path=config.checkpoint)
            print("Rank {}: checkpoint loaded.".format(self.rank))

            for layer in waveflow.sublayers():
                if isinstance(layer, weight_norm.WeightNormWrapper):
                    layer.remove_weight_norm()

            self.waveflow = waveflow

        return iteration

    def train_step(self, iteration):
        """Train the model for one step.

        Args:
            iteration (int): current iteration number.

        Returns:
            None
        """
        self.waveflow.train()

        start_time = time.time()
        audios, mels = next(self.trainloader)
        load_time = time.time()

        outputs = self.waveflow(audios, mels)
        loss = self.criterion(outputs)

        if self.parallel:
            # loss = loss / num_trainers
            loss = self.waveflow.scale_loss(loss)
            loss.backward()
            self.waveflow.apply_collective_grads()
        else:
            loss.backward()

        self.optimizer.minimize(
            loss, parameter_list=self.waveflow.parameters())
        self.waveflow.clear_gradients()

        graph_time = time.time()

        if self.rank == 0:
            loss_val = float(loss.numpy()) * self.nranks
            log = "Rank: {} Step: {:^8d} Loss: {:<8.3f} " \
                  "Time: {:.3f}/{:.3f}".format(
                  self.rank, iteration, loss_val,
                  load_time - start_time, graph_time - load_time)
            print(log)

            tb = self.tb_logger
            tb.add_scalar("Train-Loss-Rank-0", loss_val, iteration)

    @dg.no_grad
    def valid_step(self, iteration):
        """Run the model on the validation dataset.

        Args:
            iteration (int): current iteration number.

        Returns:
            None
        """
        self.waveflow.eval()
        tb = self.tb_logger

        total_loss = []
        sample_audios = []
        start_time = time.time()

        for i, batch in enumerate(self.validloader()):
            audios, mels = batch
            valid_outputs = self.waveflow(audios, mels)
            valid_z, valid_log_s_list = valid_outputs

            # Visualize latent z and scale log_s.
            if self.rank == 0 and i == 0:
                tb.add_histogram("Valid-Latent_z", valid_z.numpy(), iteration)
                for j, valid_log_s in enumerate(valid_log_s_list):
                    hist_name = "Valid-{}th-Flow-Log_s".format(j)
                    tb.add_histogram(hist_name, valid_log_s.numpy(), iteration)

            valid_loss = self.criterion(valid_outputs)
            total_loss.append(float(valid_loss.numpy()))

        total_time = time.time() - start_time
        if self.rank == 0:
            loss_val = np.mean(total_loss)
            log = "Test | Rank: {} AvgLoss: {:<8.3f} Time {:<8.3f}".format(
                self.rank, loss_val, total_time)
            print(log)
            tb.add_scalar("Valid-Avg-Loss", loss_val, iteration)

    @dg.no_grad
    def infer(self, iteration):
        """Run the model to synthesize audios.

        Args:
            iteration (int): iteration number of the loaded checkpoint.

        Returns:
            None
        """
        self.waveflow.eval()

        config = self.config
        sample = config.sample

        output = "{}/{}/iter-{}".format(config.output, config.name, iteration)
        if not os.path.exists(output):
            os.makedirs(output)

        mels_list = [mels for _, mels in self.validloader()]
        if sample is not None:
            mels_list = [mels_list[sample]]
        else:
            sample = 0

        for idx, mel in enumerate(mels_list):
            abs_idx = sample + idx
            filename = "{}/valid_{}.wav".format(output, abs_idx)
            print("Synthesize sample {}, save as {}".format(abs_idx, filename))

            start_time = time.time()
            audio = self.waveflow.synthesize(mel, sigma=self.config.sigma)
            syn_time = time.time() - start_time

            audio = audio[0]
            audio_time = audio.shape[0] / self.config.sample_rate
            print("audio time {:.4f}, synthesis time {:.4f}".format(audio_time,
                                                                    syn_time))

            # Denormalize audio from [-1, 1] to [-32768, 32768] int16 range.
            audio = audio.numpy().astype("float32") * 32768.0
            audio = audio.astype('int16')
            write(filename, config.sample_rate, audio)

    @dg.no_grad
    def benchmark(self):
        """Run the model to benchmark synthesis speed.

        Args:
            None

        Returns:
            None
        """
        self.waveflow.eval()

        mels_list = [mels for _, mels in self.validloader()]
        mel = fluid.layers.concat(mels_list, axis=2)
        mel = mel[:, :, :864]
        batch_size = 8
        mel = fluid.layers.expand(mel, [batch_size, 1, 1])

        for i in range(10):
            start_time = time.time()
            audio = self.waveflow.synthesize(mel, sigma=self.config.sigma)
            print("audio.shape = ", audio.shape)
            syn_time = time.time() - start_time

            audio_time = audio.shape[1] * batch_size / self.config.sample_rate
            print("audio time {:.4f}, synthesis time {:.4f}".format(audio_time,
                                                                    syn_time))
            print("{} X real-time".format(audio_time / syn_time))

    def save(self, iteration):
        """Save model checkpoint.

        Args:
            iteration (int): iteration number of the model to be saved.

        Returns:
            None
        """
        io.save_parameters(self.checkpoint_dir, iteration, self.waveflow,
                           self.optimizer)
