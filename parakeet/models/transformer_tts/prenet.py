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


class PreNet(dg.Layer):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(PreNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        k = math.sqrt(1 / input_size)
        self.linear1 = dg.Linear(
            input_size,
            hidden_size,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.XavierInitializer()),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-k, high=k)))
        k = math.sqrt(1 / hidden_size)
        self.linear2 = dg.Linear(
            hidden_size,
            output_size,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.XavierInitializer()),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-k, high=k)))

    def forward(self, x):
        """
        Pre Net before passing through the network.
        
        Args:
            x (Variable): The input value.
                Shape: (B, T, C), dtype: float32.
        Returns:
            (Variable), the result after pernet. Shape: (B, T, C),
        """
        x = layers.dropout(
            layers.relu(self.linear1(x)),
            self.dropout_rate,
            dropout_implementation='upscale_in_train')
        x = layers.dropout(
            layers.relu(self.linear2(x)),
            self.dropout_rate,
            dropout_implementation='upscale_in_train')
        return x
