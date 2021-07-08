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
from paddle.nn import functional as F
from paddle.fluid.layers import huber_loss

from parakeet.modules.ssim import ssim
from parakeet.modules.modules.losses import masked_l1_loss, weighted_mean
from parakeet.training.reporter import report
from parakeet.training.updaters.standard_updater import StandardUpdater
from parakeet.training.extensions.evaluator import StandardEvaluator
from parakeet.models.speedyspeech import SpeedySpeech


class SpeedySpeechUpdater(StandardUpdater):
    def update_core(self, batch):
        decoded, predicted_durations = self.model(
            text=batch["phonemes"],
            tones=batch["tones"],
            plens=batch["phoneme_lenghts"],
            durations=batch["phoneme_durations"])

        target_mel = batch["mel"]
        spec_mask = F.sequence_mask(
            batch["num_frames"], dtype=target_mel.dtype).unsqueeze(-1)
        text_mask = F.sequence_mask(
            batch["phoneme_lenghts"], dtype=predicted_durations.dtype)

        # spec loss
        l1_loss = masked_l1_loss(decoded, target_mel, spec_mask)

        # duration loss
        target_durations = batch["phoneme_durations"]
        target_durations = paddle.clip(target_durations, min=1.0)
        duration_loss = weighted_mean(
            huber_loss(
                predicted_durations, paddle.log(target_durations), delta=1.0),
            text_mask, )

        # ssim loss
        ssim_loss = 1.0 - ssim((decoded * spec_mask).unsqueeze(1),
                               (target_mel * spec_mask).unsqueeze(1))

        loss = l1_loss + duration_loss + ssim_loss

        optimizer = self.optimizer
        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        report("train/l1_loss", float(l1_loss))
        report("train/duration_loss", float(duration_loss))
        report("train/ssim_loss", float(ssim_loss))
