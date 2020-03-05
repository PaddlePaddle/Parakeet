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
import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
from parakeet.modules.customized import Conv1D
from parakeet.models.transformer_tts.utils import *
from parakeet.models.transformer_tts.cbhg import CBHG


class Vocoder(dg.Layer):
    """
    CBHG Network (mel -> linear)
    """

    def __init__(self, config, batch_size):
        super(Vocoder, self).__init__()
        self.pre_proj = Conv1D(
            num_channels=config['audio']['num_mels'],
            num_filters=config['hidden_size'],
            filter_size=1)
        self.cbhg = CBHG(config['hidden_size'], batch_size)
        self.post_proj = Conv1D(
            num_channels=config['hidden_size'],
            num_filters=(config['audio']['n_fft'] // 2) + 1,
            filter_size=1)

    def forward(self, mel):
        mel = layers.transpose(mel, [0, 2, 1])
        mel = self.pre_proj(mel)
        mel = self.cbhg(mel)
        mag_pred = self.post_proj(mel)
        mag_pred = layers.transpose(mag_pred, [0, 2, 1])
        return mag_pred
