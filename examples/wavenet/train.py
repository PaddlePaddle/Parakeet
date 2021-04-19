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

import time
from pathlib import Path
import math
import numpy as np
import paddle
from paddle import distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler
from tensorboardX import SummaryWriter
from collections import defaultdict

import parakeet
from parakeet.data import dataset
from parakeet.models.wavenet import UpsampleNet, WaveNet, ConditionalWaveNet
from parakeet.audio import AudioProcessor
from parakeet.utils import scheduler, mp_tools
from parakeet.training.cli import default_argument_parser
from parakeet.training.experiment import ExperimentBase
from parakeet.utils.mp_tools import rank_zero_only
from parakeet.datasets import AudioDataset, AudioSegmentDataset
from parakeet.data import batch_wav

from parakeet.modules.audio import STFT, MelScale

from config import get_cfg_defaults
from ljspeech import LJSpeech


class Experiment(ExperimentBase):
    def setup_model(self):
        config = self.config
        model = ConditionalWaveNet(
            upsample_factors=config.model.upsample_factors,
            n_stack=config.model.n_stack,
            n_loop=config.model.n_loop,
            residual_channels=config.model.residual_channels,
            output_dim=config.model.output_dim,
            n_mels=config.data.n_mels,
            filter_size=config.model.filter_size,
            loss_type=config.model.loss_type,
            log_scale_min=config.model.log_scale_min)

        if self.parallel:
            model = paddle.DataParallel(model)

        lr_scheduler = paddle.optimizer.lr.StepDecay(
            config.training.lr, config.training.anneal_interval,
            config.training.anneal_rate)
        optimizer = paddle.optimizer.Adam(
            lr_scheduler,
            parameters=model.parameters(),
            grad_clip=paddle.nn.ClipGradByGlobalNorm(
                config.training.gradient_max_norm))
        
        self.stft = STFT(config.data.n_fft, config.data.hop_length, config.data.win_length)
        self.mel_scale = MelScale(config.data.sample_rate, config.data.n_fft, config.data.n_mels, config.data.fmin, config.data.fmax)
        
        self.model = model
        self.model_core = model._layers if self.parallel else model
        self.optimizer = optimizer


    def setup_dataloader(self):
        config = self.config
        args = self.args

        # convolutional net's causal padding size
        context_size = config.model.n_stack \
                      * sum([(config.model.filter_size - 1) * 2**i for i in range(config.model.n_loop)]) \
                      + 1

        # frames used to compute loss
        train_clip_size = int(config.data.train_clip_seconds * config.data.sample_rate)
        length = context_size + train_clip_size
        
        root = Path(args.data).expanduser()
        file_paths = sorted(list((root / "wavs").rglob("*.wav")))
        train_set = AudioSegmentDataset(
            file_paths[config.data.valid_size:],
            config.data.sample_rate,
            length,
            top_db=config.data.top_db)
        valid_set = AudioDataset(
            file_paths[:config.data.valid_size],
            config.data.sample_rate,
            top_db=config.data.top_db)

        if not self.parallel:
            train_loader = DataLoader(
                train_set,
                batch_size=config.data.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=1,
            )
        else:
            sampler = DistributedBatchSampler(
                train_set,
                batch_size=config.data.batch_size,
                shuffle=True,
                drop_last=True)
            train_loader = DataLoader(
                train_set, batch_sampler=sampler, num_workers=1)

        valid_loader = DataLoader(
            valid_set, 
            batch_size=config.data.batch_size, 
            num_workers=1, 
            collate_fn=batch_wav)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def train_batch(self):
        # load data
        start = time.time()
        batch = self.read_batch()
        data_loader_time = time.time() - start

        self.model.train()
        self.optimizer.clear_grad()
        wav = batch
        
        # data preprocessing
        S = self.stft.magnitude(wav)
        mel = self.mel_scale(S)
        logmel = 20 * paddle.log10(mel, paddle.clip(mel, min=1e-5))
        logmel = paddle.clip((logmel + 80) / 100, min=0.0, max=1.0)
        
        # forward & backward

        y = self.model(wav, logmel)
        loss = self.model_core.loss(y, wav)
        loss.backward()
        self.optimizer.step()
        iteration_time = time.time() - start

        loss_value = float(loss)
        msg = "Rank: {}, ".format(dist.get_rank())
        msg += "step: {}, ".format(self.iteration)
        msg += "time: {:>.3f}s/{:>.3f}s, ".format(data_loader_time,
                                                  iteration_time)
        msg += "train/loss: {:>.6f}, ".format(loss_value)
        msg += "lr: {:>.6f}".format(self.optimizer.get_lr())
        self.logger.info(msg)
        if dist.get_rank() == 0:
            self.visualizer.add_scalar(
                "train/loss", loss_value, self.iteration)
            self.visualizer.add_scalar(
                "train/lr", self.optimizer.get_lr(), self.iteration)
        
        # now we have to call learning rate scheduler.step() mannually
        self.optimizer._learning_rate.step()

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def valid(self):
        valid_losses = []
        
        for batch in self.valid_loader:
            wav, length = batch
            # data preprocessing
            S = self.stft.magnitude(wav)
            mel = self.mel_scale(S)
            logmel = 20 * paddle.log10(mel, paddle.clip(mel, min=1e-5))
            logmel = paddle.clip((logmel + 80) / 100, min=0.0, max=1.0)
            
            y = self.model(wav, logmel)
            loss = self.model_core.loss(y, wav)
            valid_losses.append(float(loss))
            valid_loss = np.mean(valid_losses)
            
        msg = "Rank: {}, ".format(dist.get_rank())
        msg += "step: {}, ".format(self.iteration)
        msg += "valid/loss: {:>.6f}".format(valid_loss)
        self.logger.info(msg)
        
        self.visualizer.add_scalar(
            "valid/loss", valid_loss, self.iteration)


def main_sp(config, args):
    exp = Experiment(config, args)
    exp.setup()
    exp.run()


def main(config, args):
    if args.nprocs > 1 and args.device == "gpu":
        dist.spawn(main_sp, args=(config, args), nprocs=args.nprocs)
    else:
        main_sp(config, args)


if __name__ == "__main__":
    config = get_cfg_defaults()
    parser = default_argument_parser()
    args = parser.parse_args()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config)
    print(args)

    main(config, args)
