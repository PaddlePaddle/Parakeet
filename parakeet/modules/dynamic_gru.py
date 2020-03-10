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
import paddle.fluid.layers as layers


class DynamicGRU(dg.Layer):
    def __init__(self,
                 size,
                 param_attr=None,
                 bias_attr=None,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
                 h_0=None,
                 origin_mode=False,
                 init_size=None):
        super(DynamicGRU, self).__init__()
        self.gru_unit = dg.GRUUnit(
            size * 3,
            param_attr=param_attr,
            bias_attr=bias_attr,
            activation=candidate_activation,
            gate_activation=gate_activation,
            origin_mode=origin_mode)
        self.size = size
        self.h_0 = h_0
        self.is_reverse = is_reverse

    def forward(self, inputs):
        """
        Dynamic GRU block.
        
        Args:
            input (Variable): shape(B, T, C), dtype float32, the input value.
                 
        Returns:
            output (Variable): shape(B, T, C), the result compute by GRU.
        """
        hidden = self.h_0
        res = []
        for i in range(inputs.shape[1]):
            if self.is_reverse:
                i = inputs.shape[1] - 1 - i
            input_ = inputs[:, i:i + 1, :]
            input_ = layers.reshape(input_, [-1, input_.shape[2]])
            hidden, reset, gate = self.gru_unit(input_, hidden)
            hidden_ = layers.reshape(hidden, [-1, 1, hidden.shape[1]])
            res.append(hidden_)
        if self.is_reverse:
            res = res[::-1]
        res = layers.concat(res, axis=1)
        return res
