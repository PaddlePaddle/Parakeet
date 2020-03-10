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
import paddle.fluid as fluid
import math
from parakeet.modules.customized import Conv1D


class PositionwiseFeedForward(dg.Layer):
    def __init__(self,
                 d_in,
                 num_hidden,
                 filter_size,
                 padding=0,
                 use_cudnn=True,
                 dropout=0.1):
        """A two-feed-forward-layer module.

        Args:
            d_in (int): the size of input channel.
            num_hidden (int): the size of hidden layer in network.
            filter_size (int): the filter size of Conv
            padding (int, optional): the padding size of Conv. Defaults to 0.
            use_cudnn (bool, optional): use cudnn in Conv or not. Defaults to True.
            dropout (float, optional): dropout probability. Defaults to 0.1.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.num_hidden = num_hidden
        self.use_cudnn = use_cudnn
        self.dropout = dropout

        k = math.sqrt(1.0 / d_in)
        self.w_1 = Conv1D(
            num_channels=d_in,
            num_filters=num_hidden,
            filter_size=filter_size,
            padding=padding,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.XavierInitializer()),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-k, high=k)),
            use_cudnn=use_cudnn)
        k = math.sqrt(1.0 / num_hidden)
        self.w_2 = Conv1D(
            num_channels=num_hidden,
            num_filters=d_in,
            filter_size=filter_size,
            padding=padding,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.XavierInitializer()),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-k, high=k)),
            use_cudnn=use_cudnn)
        self.layer_norm = dg.LayerNorm(d_in)

    def forward(self, input):
        """
        Compute feed forward network result.
        
        Args:
            input (Variable): shape(B, T, C), dtype float32, the input value. 
                
        Returns:
            output (Variable): shape(B, T, C), the result after FFN. 
        """
        x = layers.transpose(input, [0, 2, 1])
        #FFN Networt
        x = self.w_2(layers.relu(self.w_1(x)))

        # dropout
        x = layers.dropout(
            x, self.dropout, dropout_implementation='upscale_in_train')

        x = layers.transpose(x, [0, 2, 1])
        # residual connection
        x = x + input

        #layer normalization
        output = self.layer_norm(x)

        return output
