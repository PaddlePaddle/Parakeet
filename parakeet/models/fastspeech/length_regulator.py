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
import numpy as np
import math
import parakeet.models.fastspeech.utils
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers
import paddle.fluid as fluid
from parakeet.modules.customized import Conv1D


class LengthRegulator(dg.Layer):
    def __init__(self, input_size, out_channels, filter_size, dropout=0.1):
        """Length Regulator block in FastSpeech.

        Args:
            input_size (int): the channel number of input.
            out_channels (int): the output channel number.
            filter_size (int): the filter size of duration predictor.
            dropout (float, optional): dropout probability. Defaults to 0.1.
        """
        super(LengthRegulator, self).__init__()
        self.duration_predictor = DurationPredictor(
            input_size=input_size,
            out_channels=out_channels,
            filter_size=filter_size,
            dropout=dropout)

    def LR(self, x, duration_predictor_output, alpha=1.0):
        output = []
        batch_size = x.shape[0]
        for i in range(batch_size):
            output.append(
                self.expand(x[i:i + 1], duration_predictor_output[i:i + 1],
                            alpha))
        output = self.pad(output)
        return output

    def pad(self, input_ele):
        max_len = max([input_ele[i].shape[0] for i in range(len(input_ele))])
        out_list = []
        for i in range(len(input_ele)):
            pad_len = max_len - input_ele[i].shape[0]
            one_batch_padded = layers.pad(input_ele[i], [0, pad_len, 0, 0],
                                          pad_value=0.0)
            out_list.append(one_batch_padded)
        out_padded = layers.stack(out_list)
        return out_padded

    def expand(self, batch, predicted, alpha):
        out = []
        time_steps = batch.shape[1]
        fertilities = predicted.numpy()
        batch = layers.squeeze(batch, [0])

        for i in range(time_steps):
            if fertilities[0, i] == 0:
                continue
            out.append(
                layers.expand(batch[i:i + 1, :], [int(fertilities[0, i]), 1]))
        out = layers.concat(out, axis=0)
        return out

    def forward(self, x, alpha=1.0, target=None):
        """
        Compute length of mel from encoder output use TransformerTTS attention
        
        Args:
            x (Variable): shape(B, T, C), dtype float32, the encoder output.
            alpha (float32, optional): the hyperparameter to determine the length of 
                the expanded sequence mel, thereby controlling the voice speed. Defaults to 1.0.
            target (Variable, optional): shape(B, T_text), dtype int64, the duration of phoneme compute from pretrained transformerTTS. 
                Defaults to None. 

        Returns:
            output (Variable): shape(B, T, C), the output after exppand.
            duration_predictor_output (Variable): shape(B, T, C), the output of duration predictor.
        """
        duration_predictor_output = self.duration_predictor(x)
        if fluid.framework._dygraph_tracer()._train_mode:
            output = self.LR(x, target)
            return output, duration_predictor_output
        else:
            duration_predictor_output = layers.round(duration_predictor_output)
            output = self.LR(x, duration_predictor_output, alpha)
            mel_pos = dg.to_variable(np.arange(1, output.shape[1] + 1))
            mel_pos = layers.unsqueeze(mel_pos, [0])
            return output, mel_pos


class DurationPredictor(dg.Layer):
    def __init__(self, input_size, out_channels, filter_size, dropout=0.1):
        """Duration Predictor block in FastSpeech.

        Args:
            input_size (int): the channel number of input.
            out_channels (int): the output channel number.
            filter_size (int): the filter size.
            dropout (float, optional): dropout probability. Defaults to 0.1.
        """
        super(DurationPredictor, self).__init__()
        self.input_size = input_size
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.dropout = dropout

        k = math.sqrt(1.0 / self.input_size)
        self.conv1 = Conv1D(
            num_channels=self.input_size,
            num_filters=self.out_channels,
            filter_size=self.filter_size,
            padding=1,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.XavierInitializer()),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-k, high=k)))
        #data_format='NTC')
        k = math.sqrt(1.0 / self.out_channels)
        self.conv2 = Conv1D(
            num_channels=self.out_channels,
            num_filters=self.out_channels,
            filter_size=self.filter_size,
            padding=1,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.XavierInitializer()),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-k, high=k)))
        #data_format='NTC')
        self.layer_norm1 = dg.LayerNorm(self.out_channels)
        self.layer_norm2 = dg.LayerNorm(self.out_channels)

        self.weight = fluid.ParamAttr(
            initializer=fluid.initializer.XavierInitializer())
        k = math.sqrt(1.0 / self.out_channels)
        self.bias = fluid.ParamAttr(initializer=fluid.initializer.Uniform(
            low=-k, high=k))

        self.linear = dg.Linear(
            self.out_channels, 1, param_attr=self.weight, bias_attr=self.bias)

    def forward(self, encoder_output):
        """
        Predict the duration of each character.
        
        Args:
            encoder_output (Variable): shape(B, T, C), dtype float32, the encoder output.
        
        Returns:
            out (Variable): shape(B, T, C), the output of duration predictor.
        """
        # encoder_output.shape(N, T, C)
        out = layers.transpose(encoder_output, [0, 2, 1])
        out = self.conv1(out)
        out = layers.transpose(out, [0, 2, 1])
        out = layers.dropout(
            layers.relu(self.layer_norm1(out)),
            self.dropout,
            dropout_implementation='upscale_in_train')
        out = layers.transpose(out, [0, 2, 1])
        out = self.conv2(out)
        out = layers.transpose(out, [0, 2, 1])
        out = layers.dropout(
            layers.relu(self.layer_norm2(out)),
            self.dropout,
            dropout_implementation='upscale_in_train')
        out = layers.relu(self.linear(out))
        out = layers.squeeze(out, axes=[-1])

        return out
