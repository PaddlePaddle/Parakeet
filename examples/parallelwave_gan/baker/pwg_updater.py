# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from parakeet.datasets.data_table import DataTable
from parakeet.training.updater import UpdaterBase, UpdaterState
from parakeet.training.trainer import Trainer
from parakeet.training.reporter import report
from parakeet.training.checkpoint import KBest, KLatest
from parakeet.models.parallel_wavegan import PWGGenerator, PWGDiscriminator
from parakeet.modules.stft_loss import MultiResolutionSTFTLoss


class PWGUpdater(UpdaterBase):
    def __init__(
            self,
            models,
            optimizers,
            criterions,
            schedulers,
            dataloaders,
            discriminator_train_start_steps: int,
            lambda_adv: float, ):
        self.models = models
        self.generator = models['generator']
        self.discriminator = models['discriminator']

        self.optimizers = optimizers
        self.optimizer_g = optimizers['generator']
        self.optimizer_d = optimizers['discriminator']

        self.criterions = criterions
        self.criterion_stft = criterions['stft']
        self.criterion_mse = criterions['mse']

        self.schedulers = schedulers
        self.scheduler_g = schedulers['generator']
        self.scheduler_d = schedulers['discriminator']

        self.dataloaders = dataloaders
        self.train_dataloader = dataloaders['train']
        self.dev_dataloader = dataloaders['dev']

        self.discriminator_train_start_steps = discriminator_train_start_steps
        self.lambda_adv = lambda_adv
        self.state = UpdaterState(iteration=0, epoch=0)

        self.train_iterator = iter(self.train_dataloader)

    def update_core(self):
        try:
            batch = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_dataloader)
            batch = next(self.train_iterator)

        wav, mel = batch

        # Generator
        noise = paddle.randn(wav.shape)
        wav_ = self.generator(noise, mel)

        ## Multi-resolution stft loss
        sc_loss, mag_loss = self.criterion_stft(
            wav_.squeeze(1), wav.squeeze(1))
        report("train/spectral_convergence_loss", float(sc_loss))
        report("train/log_stft_magnitude_loss", float(mag_loss))
        gen_loss = sc_loss + mag_loss

        ## Adversarial loss
        if self.state.iteration > self.discriminator_train_start_steps:
            p_ = self.discriminator(wav_)
            adv_loss = self.criterion_mse(p_, paddle.ones_like(p_))
            report("train/adversarial_loss", float(adv_loss))
            gen_loss += self.lambda_adv * adv_loss

        report("train/generator_loss", float(gen_loss))
        self.optimizer_g.clear_grad()
        gen_loss.backward()
        self.optimizer_g.step()
        self.scheduler_g.step()

        # Disctiminator
        if self.state.iteration > self.discriminator_train_start_steps:
            with paddle.no_grad():
                wav_ = self.generator(noise, mel)
            p = self.discriminator(wav)
            p_ = self.discriminator(wav_.detach())
            real_loss = self.criterion_mse(p, paddle.ones_like(p))
            fake_loss = self.criterion_mse(p_, paddle.zeros_like(p_))
            report("train/real_loss", float(real_loss))
            report("train/fake_loss", float(fake_loss))
            dis_loss = real_loss + fake_loss
            report("train/discriminator_loss", float(dis_loss))

            self.optimizer_d.clear_grad()
            dis_loss.backward()
            self.optimizer_d.step()
            self.scheduler_d.step()
