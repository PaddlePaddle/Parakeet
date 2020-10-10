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
from parakeet.g2p.text.symbols import symbols
import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from parakeet.modules.customized import Conv1D
import numpy as np


class EncoderPrenet(dg.Layer):
    def __init__(self, embedding_size, num_hidden, use_cudnn=True):
        """ Encoder prenet layer of TransformerTTS.

        Args:
            embedding_size (int): the size of embedding.
            num_hidden (int): the size of hidden layer in network.
            use_cudnn (bool, optional): use cudnn or not. Defaults to True.
        """
        super(EncoderPrenet, self).__init__()
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden
        self.use_cudnn = use_cudnn
        self.embedding = dg.Embedding(
            size=[len(symbols), embedding_size],
            padding_idx=0,
            param_attr=fluid.initializer.Normal(
                loc=0.0, scale=1.0))
        self.conv_list = []
        k = math.sqrt(1.0 / embedding_size)
        self.conv_list.append(
            Conv1D(
                num_channels=embedding_size,
                num_filters=num_hidden,
                filter_size=5,
                padding=int(np.floor(5 / 2)),
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.XavierInitializer()),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(
                        low=-k, high=k)),
                use_cudnn=use_cudnn))
        k = math.sqrt(1.0 / num_hidden)
        for _ in range(2):
            self.conv_list.append(
                Conv1D(
                    num_channels=num_hidden,
                    num_filters=num_hidden,
                    filter_size=5,
                    padding=int(np.floor(5 / 2)),
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
                num_hidden, data_layout='NCHW') for _ in range(3)
        ]

        for i, layer in enumerate(self.batch_norm_list):
            self.add_sublayer("batch_norm_list_{}".format(i), layer)

        k = math.sqrt(1.0 / num_hidden)
        self.projection = dg.Linear(
            num_hidden,
            num_hidden,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.XavierInitializer()),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-k, high=k)))

    def forward(self, x):
        """
        Prepare encoder input.
        
        Args:
            x (Variable): shape(B, T_text), dtype float32, the input character, where T_text means the timesteps of input text.
                
        Returns:
            (Variable): shape(B, T_text, C), the encoder prenet output.
        """

        x = self.embedding(x)
        x = layers.transpose(x, [0, 2, 1])
        for batch_norm, conv in zip(self.batch_norm_list, self.conv_list):
            x = layers.dropout(
                layers.relu(batch_norm(conv(x))),
                0.2,
                dropout_implementation='upscale_in_train')
        x = layers.transpose(x, [0, 2, 1])  #(N,T,C)
        x = self.projection(x)

        return x
