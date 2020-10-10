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
    def __init__(self, batch_size, hidden_size, num_mels=80, n_fft=2048):
        """CBHG Network (mel -> linear)

        Args:
            batch_size (int): the batch size of input.
            hidden_size (int): the size of hidden layer in network.
            n_mels (int, optional): the number of mel bands when calculating mel spectrograms. Defaults to 80.
            n_fft (int, optional): length of the windowed signal after padding with zeros. Defaults to 2048.
        """
        super(Vocoder, self).__init__()
        self.pre_proj = Conv1D(
            num_channels=num_mels, num_filters=hidden_size, filter_size=1)
        self.cbhg = CBHG(hidden_size, batch_size)
        self.post_proj = Conv1D(
            num_channels=hidden_size,
            num_filters=(n_fft // 2) + 1,
            filter_size=1)

    def forward(self, mel):
        """
        Compute mel spectrum to linear spectrum.
        
        Args:
            mel (Variable): shape(B, C, T), dtype float32, the input mel spectrum.
                
        Returns:
            mag_pred (Variable): shape(B, T, C), the linear output.
        """
        mel = layers.transpose(mel, [0, 2, 1])
        mel = self.pre_proj(mel)
        mel = self.cbhg(mel)
        mag_pred = self.post_proj(mel)
        mag_pred = layers.transpose(mag_pred, [0, 2, 1])
        return mag_pred
