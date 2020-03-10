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
from parakeet.modules.customized import Pool1D, Conv1D
from parakeet.modules.dynamic_gru import DynamicGRU
import numpy as np


class CBHG(dg.Layer):
    def __init__(self,
                 hidden_size,
                 batch_size,
                 K=16,
                 projection_size=256,
                 num_gru_layers=2,
                 max_pool_kernel_size=2,
                 is_post=False):
        """CBHG Module

        Args:
            hidden_size (int): dimension of hidden unit.
            batch_size (int): batch size of input.
            K (int, optional): number of convolution banks. Defaults to 16.
            projection_size (int, optional): dimension of projection unit. Defaults to 256.
            num_gru_layers (int, optional): number of layers of GRUcell. Defaults to 2.
            max_pool_kernel_size (int, optional): max pooling kernel size. Defaults to 2
            is_post (bool, optional): whether post processing or not. Defaults to False.
        """
        super(CBHG, self).__init__()

        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.conv_list = []
        k = math.sqrt(1.0 / projection_size)
        self.conv_list.append(
            Conv1D(
                num_channels=projection_size,
                num_filters=hidden_size,
                filter_size=1,
                padding=int(np.floor(1 / 2)),
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.XavierInitializer()),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Uniform(
                        low=-k, high=k))))
        k = math.sqrt(1.0 / hidden_size)
        for i in range(2, K + 1):
            self.conv_list.append(
                Conv1D(
                    num_channels=hidden_size,
                    num_filters=hidden_size,
                    filter_size=i,
                    padding=int(np.floor(i / 2)),
                    param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.XavierInitializer()),
                    bias_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Uniform(
                            low=-k, high=k))))

        for i, layer in enumerate(self.conv_list):
            self.add_sublayer("conv_list_{}".format(i), layer)

        self.batchnorm_list = []
        for i in range(K):
            self.batchnorm_list.append(
                dg.BatchNorm(
                    hidden_size, data_layout='NCHW'))

        for i, layer in enumerate(self.batchnorm_list):
            self.add_sublayer("batchnorm_list_{}".format(i), layer)

        conv_outdim = hidden_size * K

        k = math.sqrt(1.0 / conv_outdim)
        self.conv_projection_1 = Conv1D(
            num_channels=conv_outdim,
            num_filters=hidden_size,
            filter_size=3,
            padding=int(np.floor(3 / 2)),
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.XavierInitializer()),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-k, high=k)))

        k = math.sqrt(1.0 / hidden_size)
        self.conv_projection_2 = Conv1D(
            num_channels=hidden_size,
            num_filters=projection_size,
            filter_size=3,
            padding=int(np.floor(3 / 2)),
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.XavierInitializer()),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-k, high=k)))

        self.batchnorm_proj_1 = dg.BatchNorm(hidden_size, data_layout='NCHW')
        self.batchnorm_proj_2 = dg.BatchNorm(
            projection_size, data_layout='NCHW')
        self.max_pool = Pool1D(
            pool_size=max_pool_kernel_size,
            pool_type='max',
            pool_stride=1,
            pool_padding=1,
            data_format="NCT")
        self.highway = Highwaynet(self.projection_size)

        h_0 = np.zeros((batch_size, hidden_size // 2), dtype="float32")
        h_0 = dg.to_variable(h_0)
        k = math.sqrt(1.0 / hidden_size)
        self.fc_forward1 = dg.Linear(
            hidden_size,
            hidden_size // 2 * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.XavierInitializer()),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-k, high=k)))
        self.fc_reverse1 = dg.Linear(
            hidden_size,
            hidden_size // 2 * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.XavierInitializer()),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-k, high=k)))
        self.gru_forward1 = DynamicGRU(
            size=self.hidden_size // 2,
            is_reverse=False,
            origin_mode=True,
            h_0=h_0)
        self.gru_reverse1 = DynamicGRU(
            size=self.hidden_size // 2,
            is_reverse=True,
            origin_mode=True,
            h_0=h_0)

        self.fc_forward2 = dg.Linear(
            hidden_size,
            hidden_size // 2 * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.XavierInitializer()),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-k, high=k)))
        self.fc_reverse2 = dg.Linear(
            hidden_size,
            hidden_size // 2 * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.XavierInitializer()),
            bias_attr=fluid.ParamAttr(initializer=fluid.initializer.Uniform(
                low=-k, high=k)))
        self.gru_forward2 = DynamicGRU(
            size=self.hidden_size // 2,
            is_reverse=False,
            origin_mode=True,
            h_0=h_0)
        self.gru_reverse2 = DynamicGRU(
            size=self.hidden_size // 2,
            is_reverse=True,
            origin_mode=True,
            h_0=h_0)

    def _conv_fit_dim(self, x, filter_size=3):
        if filter_size % 2 == 0:
            return x[:, :, :-1]
        else:
            return x

    def forward(self, input_):
        """
        Convert linear spectrum to Mel spectrum.

        Args:
            input_ (Variable): shape(B, C, T), dtype float32, the sequentially input.  

        Returns:
            out (Variable): shape(B, C, T), the CBHG output.
        """

        conv_list = []
        conv_input = input_

        for i, (conv, batchnorm
                ) in enumerate(zip(self.conv_list, self.batchnorm_list)):
            conv_input = self._conv_fit_dim(conv(conv_input), i + 1)
            conv_input = layers.relu(batchnorm(conv_input))
            conv_list.append(conv_input)

        conv_cat = layers.concat(conv_list, axis=1)
        conv_pool = self.max_pool(conv_cat)[:, :, :-1]

        conv_proj = layers.relu(
            self.batchnorm_proj_1(
                self._conv_fit_dim(self.conv_projection_1(conv_pool))))
        conv_proj = self.batchnorm_proj_2(
            self._conv_fit_dim(self.conv_projection_2(conv_proj))) + input_

        # conv_proj.shape = [N, C, T]
        highway = layers.transpose(conv_proj, [0, 2, 1])
        highway = self.highway(highway)

        # highway.shape = [N, T, C]
        fc_forward = self.fc_forward1(highway)
        fc_reverse = self.fc_reverse1(highway)
        out_forward = self.gru_forward1(fc_forward)
        out_reverse = self.gru_reverse1(fc_reverse)
        out = layers.concat([out_forward, out_reverse], axis=-1)
        fc_forward = self.fc_forward2(out)
        fc_reverse = self.fc_reverse2(out)
        out_forward = self.gru_forward2(fc_forward)
        out_reverse = self.gru_reverse2(fc_reverse)
        out = layers.concat([out_forward, out_reverse], axis=-1)
        out = layers.transpose(out, [0, 2, 1])
        return out


class Highwaynet(dg.Layer):
    def __init__(self, num_units, num_layers=4):
        """Highway network

        Args:
            num_units (int): dimension of hidden unit.
            num_layers (int, optional): number of highway layers. Defaults to 4.
        """
        super(Highwaynet, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers

        self.gates = []
        self.linears = []
        k = math.sqrt(1.0 / num_units)
        for i in range(num_layers):
            self.linears.append(
                dg.Linear(
                    num_units,
                    num_units,
                    param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.XavierInitializer()),
                    bias_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Uniform(
                            low=-k, high=k))))
            self.gates.append(
                dg.Linear(
                    num_units,
                    num_units,
                    param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.XavierInitializer()),
                    bias_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.Uniform(
                            low=-k, high=k))))

        for i, (linear, gate) in enumerate(zip(self.linears, self.gates)):
            self.add_sublayer("linears_{}".format(i), linear)
            self.add_sublayer("gates_{}".format(i), gate)

    def forward(self, input_):
        """
        Compute result of Highway network.

        Args:
            input_(Variable): shape(B, T, C), dtype float32, the sequentially input.
            
        Returns:
            out(Variable): the Highway output.
        """
        out = input_

        for linear, gate in zip(self.linears, self.gates):
            h = fluid.layers.relu(linear(out))
            t_ = fluid.layers.sigmoid(gate(out))

            c = 1 - t_
            out = h * t_ + out * c

        return out
