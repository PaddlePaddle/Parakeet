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
import math
import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from parakeet.modules.customized import Conv1D


class PostConvNet(dg.Layer):
    def __init__(self,
                 n_mels=80,
                 num_hidden=512,
                 filter_size=5,
                 padding=0,
                 num_conv=5,
                 outputs_per_step=1,
                 use_cudnn=True,
                 dropout=0.1,
                 batchnorm_last=False):
        super(PostConvNet, self).__init__()

        self.dropout = dropout
        self.num_conv = num_conv
        self.batchnorm_last = batchnorm_last
        self.conv_list = []
        k = math.sqrt(1 / (n_mels * outputs_per_step))
        self.conv_list.append(
            Conv1D(
                num_channels=n_mels * outputs_per_step,
                num_filters=num_hidden,
                filter_size=filter_size,
                padding=padding,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.XavierInitializer()),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(
                        low=-k, high=k)),
                use_cudnn=use_cudnn))

        k = math.sqrt(1 / num_hidden)
        for _ in range(1, num_conv - 1):
            self.conv_list.append(
                Conv1D(
                    num_channels=num_hidden,
                    num_filters=num_hidden,
                    filter_size=filter_size,
                    padding=padding,
                    param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.XavierInitializer()),
                    bias_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Uniform(
                            low=-k, high=k)),
                    use_cudnn=use_cudnn))

        self.conv_list.append(
            Conv1D(
                num_channels=num_hidden,
                num_filters=n_mels * outputs_per_step,
                filter_size=filter_size,
                padding=padding,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.XavierInitializer()),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(
                        low=-k, high=k)),
                use_cudnn=use_cudnn))

        for i, layer in enumerate(self.conv_list):
            self.add_sublayer("conv_list_{}".format(i), layer)

        self.batch_norm_list = [
            dg.BatchNorm(
                num_hidden, data_layout='NCHW') for _ in range(num_conv - 1)
        ]
        if self.batchnorm_last:
            self.batch_norm_list.append(
                dg.BatchNorm(
                    n_mels * outputs_per_step, data_layout='NCHW'))
        for i, layer in enumerate(self.batch_norm_list):
            self.add_sublayer("batch_norm_list_{}".format(i), layer)

    def forward(self, input):
        """
        Post Conv Net.
        
        Args:
            input (Variable): Shape(B, T, C), dtype: float32. The input value.
        Returns:
            output (Variable), Shape(B, T, C), the result after postconvnet.
        """

        input = layers.transpose(input, [0, 2, 1])
        len = input.shape[-1]
        for i in range(self.num_conv - 1):
            batch_norm = self.batch_norm_list[i]
            conv = self.conv_list[i]

            input = layers.dropout(
                layers.tanh(batch_norm(conv(input)[:, :, :len])), self.dropout)
        conv = self.conv_list[self.num_conv - 1]
        input = conv(input)[:, :, :len]
        if self.batchnorm_last:
            batch_norm = self.batch_norm_list[self.num_conv - 1]
            input = layers.dropout(batch_norm(input), self.dropout)
        output = layers.transpose(input, [0, 2, 1])
        return output
