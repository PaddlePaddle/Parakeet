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

import math
import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F


class Stretch2D(nn.Layer):
    def __init__(self, x_scale, y_scale, mode="nearest"):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        out = F.interpolate(
            x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode)
        return out


class UpsampleNet(nn.Layer):
    def __init__(self,
                 upsample_scales,
                 nonlinear_activation=None,
                 nonlinear_activation_params={},
                 interpolate_mode="nearest",
                 freq_axis_kernel_size=1,
                 use_causal_conv=False):
        super().__init__()
        self.use_causal_conv = use_causal_conv
        self.up_layers = nn.LayerList()
        for scale in upsample_scales:
            stretch = Stretch2D(scale, 1, interpolate_mode)
            assert freq_axis_kernel_size % 2 == 1
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            kernel_size = (freq_axis_kernel_size, scale * 2 + 1)
            if use_causal_conv:
                padding = (freq_axis_padding, scale * 2)
            else:
                padding = (freq_axis_padding, scale)
            conv = nn.Conv2D(
                1, 1, kernel_size, padding=padding, bias_attr=False)
            if nonlinear_activation is not None:
                nonlinear = getattr(
                    nn, nonlinear_activation)(**nonlinear_activation_params)
            self.up_layers.extend([stretch, conv, nonlinear])

    def forward(self, c):
        c = c.unsqueeze(1)
        for f in self.up_layers:
            if self.use_causal_conv and isinstance(f, nn.Conv2D):
                c = f(c)[:, :, :, c.shape[-1]]
            else:
                c = f(c)
        return c.squeeze(1)


class ConvInUpsampleNet(nn.Layer):
    def __init__(self,
                 upsample_scales,
                 nonlinear_activation=None,
                 nonlinear_activation_params={},
                 interpolate_mode="nearest",
                 freq_axis_kernel_size=1,
                 aux_channels=80,
                 aux_context_window=0,
                 use_causal_conv=False):
        super().__init__()
        self.aux_context_window = aux_context_window
        self.use_causal_conv = use_causal_conv and aux_context_window > 0
        kernel_size = aux_context_window + 1 if use_causal_conv else 2 * aux_context_window + 1
        self.conv_in = nn.Conv1D(
            aux_channels,
            aux_channels,
            kernel_size=kernel_size,
            bias_attr=False)
        self.upsample = UpsampleNet(
            upsample_scales=upsample_scales,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            interpolate_mode=interpolate_mode,
            freq_axis_kernel_size=freq_axis_kernel_size,
            use_causal_conv=use_causal_conv)

    def forward(self, c):
        c_ = self.conv_in(c)
        c = c_[:, :, :-self.aux_context_window] if self.use_causal_conv else c_
        return self.upsample(c)


class ResidualBlock(nn.Layer):
    def __init__(self,
                 kernel_size=3,
                 residual_channels=64,
                 gate_channels=128,
                 skip_channels=64,
                 aux_channels=80,
                 dropout=0.,
                 dilation=1,
                 bias=True,
                 use_causal_conv=False):
        super().__init__()
        self.dropout = dropout
        if use_causal_conv:
            padding = (kernel_size - 1) * dilation
        else:
            assert kernel_size % 2 == 1
            padding = (kernel_size - 1) // 2 * dilation
        self.use_causal_conv = use_causal_conv

        self.conv = nn.Conv1D(
            residual_channels,
            gate_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias_attr=bias)
        if aux_channels is not None:
            self.conv1x1_aux = nn.Conv1D(
                aux_channels, gate_channels, kernel_size=1, bias_attr=False)
        else:
            self.conv1x1_aux = None

        gate_out_channels = gate_channels // 2
        self.conv1x1_out = nn.Conv1D(
            gate_out_channels,
            residual_channels,
            kernel_size=1,
            bias_attr=bias)
        self.conv1x1_skip = nn.Conv1D(
            gate_out_channels, skip_channels, kernel_size=1, bias_attr=bias)

    def forward(self, x, c):
        x_input = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv(x)
        x = x[:, :, x_input.shape[-1]] if self.use_causal_conv else x
        if c is not None:
            c = self.conv1x1_aux(c)
            x += c

        a, b = paddle.chunk(x, 2, axis=1)
        x = paddle.tanh(a) * F.sigmoid(b)

        skip = self.conv1x1_skip(x)
        res = (self.conv1x1_out(x) + x_input) * math.sqrt(0.5)
        return res, skip


class PWGGenerator(nn.Layer):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 layers=30,
                 stacks=3,
                 residual_channels=64,
                 gate_channels=128,
                 skip_channels=64,
                 aux_channels=80,
                 aux_context_window=2,
                 dropout=0.,
                 bias=True,
                 use_weight_norm=True,
                 use_causal_conv=False,
                 upsample_scales=[4, 4, 4, 4],
                 nonlinear_activation=None,
                 nonlinear_activation_params={},
                 interpolate_mode="nearest",
                 freq_axis_kernel_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.aux_context_window = aux_context_window
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size

        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        self.first_conv = nn.Conv1D(
            in_channels, residual_channels, 1, bias_attr=True)
        self.upsample_net = ConvInUpsampleNet(
            upsample_scales=upsample_scales,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            interpolate_mode=interpolate_mode,
            freq_axis_kernel_size=freq_axis_kernel_size,
            aux_channels=aux_channels,
            aux_context_window=aux_context_window,
            use_causal_conv=use_causal_conv)
        self.upsample_factor = np.prod(upsample_scales)

        self.conv_layers = nn.LayerList()
        for layer in range(layers):
            dilation = 2**(layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
                use_causal_conv=use_causal_conv)
            self.conv_layers.append(conv)

        self.last_conv_layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1D(
                skip_channels, skip_channels, 1, bias_attr=True),
            nn.ReLU(),
            nn.Conv1D(
                skip_channels, out_channels, 1, bias_attr=True))

        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, c):
        if c is not None:
            c = self.upsample_net(c)
            assert c.shape[-1] == x.shape[-1]

        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, s = f(x, c)
            skips += s
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        x = self.last_conv_layers(skips)
        return x

    def apply_weight_norm(self):
        def _apply_weight_norm(layer):
            if isinstance(layer, (nn.Conv1D, nn.Conv2D)):
                nn.utils.weight_norm(layer)

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        def _remove_weight_norm(layer):
            try:
                nn.utils.remove_weight_norm(layer)
            except ValueError:
                pass

        self.apply(_remove_weight_norm)

    def inference(self, c=None, x=None):
        """
        single instance inference
        c: [T', C] condition
        x: [T, 1] noise
        """
        if x is not None:
            x = paddle.transpose(x, [1, 0]).unsqueeze(0)  # pseudo batch
        else:
            assert c is not None
            x = paddle.randn([1, 1, c.shape[0] * self.upsample_factor])

        if c is not None:
            c = paddle.transpose(c, [1, 0]).unsqueeze(0)  # pseudo batch
            c = nn.Pad1D(self.aux_context_window, mode='edge')(c)
        out = self.forward(x, c).squeeze(0).transpose([1, 0])
        return out


class PWGDiscriminator(nn.Layer):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 layers=10,
                 conv_channels=64,
                 dilation_factor=1,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 bias=True,
                 use_weight_norm=True):
        super().__init__()
        assert kernel_size % 2 == 1
        assert dilation_factor > 0
        conv_layers = []
        conv_in_channels = in_channels
        for i in range(layers - 1):
            if i == 0:
                dilation = 1
            else:
                dilation = i if dilation_factor == 1 else dilation_factor**i
                conv_in_channels = conv_channels
            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = nn.Conv1D(
                conv_in_channels,
                conv_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias_attr=bias)
            nonlinear = getattr(
                nn, nonlinear_activation)(**nonlinear_activation_params)
            conv_layers.append(conv_layer)
            conv_layers.append(nonlinear)
        padding = (kernel_size - 1) // 2
        last_conv = nn.Conv1D(
            conv_in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias_attr=bias)
        conv_layers.append(last_conv)
        self.conv_layers = nn.Sequential(*conv_layers)

        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        return self.conv_layers(x)

    def apply_weight_norm(self):
        def _apply_weight_norm(layer):
            if isinstance(layer, (nn.Conv1D, nn.Conv2D)):
                nn.utils.weight_norm(layer)

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        def _remove_weight_norm(layer):
            try:
                nn.utils.remove_weight_norm(layer)
            except ValueError:
                pass

        self.apply(_remove_weight_norm)


class ResidualPWGDiscriminator(nn.Layer):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 layers=30,
                 stacks=3,
                 residual_channels=64,
                 gate_channels=128,
                 skip_channels=64,
                 dropout=0.,
                 bias=True,
                 use_weight_norm=True,
                 use_causal_conv=False,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2}):
        super().__init__()
        assert kernel_size % 2 == 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size

        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        self.first_conv = nn.Sequential(
            nn.Conv1D(
                in_channels, residual_channels, 1, bias_attr=True),
            getattr(nn, nonlinear_activation)(**nonlinear_activation_params))

        self.conv_layers = nn.LayerList()
        for layer in range(layers):
            dilation = 2**(layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=None,  # no auxiliary input
                dropout=dropout,
                dilation=dilation,
                bias=bias,
                use_causal_conv=use_causal_conv)
            self.conv_layers.append(conv)

        self.last_conv_layers = nn.Sequential(
            getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
            nn.Conv1D(
                skip_channels, skip_channels, 1, bias_attr=True),
            getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
            nn.Conv1D(
                skip_channels, out_channels, 1, bias_attr=True))

        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        x = self.first_conv(x)
        skip = 0
        for f in self.conv_layers:
            x, h = f(x, None)
            skip += h
        skip *= math.sqrt(1 / len(self.conv_layers))

        x = skip
        x = self.last_conv_layers(x)
        return x

    def apply_weight_norm(self):
        def _apply_weight_norm(layer):
            if isinstance(layer, (nn.Conv1D, nn.Conv2D)):
                nn.utils.weight_norm(layer)

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        def _remove_weight_norm(layer):
            try:
                nn.utils.remove_weight_norm(layer)
            except ValueError:
                pass

        self.apply(_remove_weight_norm)
