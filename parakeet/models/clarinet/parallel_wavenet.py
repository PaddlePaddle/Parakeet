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

from __future__ import division
import math
import time
import itertools
import numpy as np

import paddle.fluid.layers as F
import paddle.fluid.dygraph as dg
import paddle.fluid.initializer as I
import paddle.fluid.layers.distributions as D

from parakeet.modules.weight_norm import Linear, Conv1D, Conv1DCell, Conv2DTranspose
from parakeet.models.wavenet import WaveNet


class ParallelWaveNet(dg.Layer):
    def __init__(self, n_loops, n_layers, residual_channels, condition_dim,
                 filter_size):
        """ParallelWaveNet, an inverse autoregressive flow model, it contains several flows(WaveNets).

        Args:
            n_loops (List[int]): `n_loop` for each flow.
            n_layers (List[int]): `n_layer` for each flow.
            residual_channels (int): `residual_channels` for every flow.
            condition_dim (int): `condition_dim` for every flow.
            filter_size (int): `filter_size` for every flow.
        """
        super(ParallelWaveNet, self).__init__()
        self.flows = dg.LayerList()
        for n_loop, n_layer in zip(n_loops, n_layers):
            # teacher's log_scale_min does not matter herem, -100 is a dummy value
            self.flows.append(
                WaveNet(n_loop, n_layer, residual_channels, 3, condition_dim,
                        filter_size, "mog", -100.0))

    def forward(self, z, condition=None):
        """Transform a random noise sampled from a standard Gaussian distribution into sample from the target distribution. And output the mean and log standard deviation of the output distribution.

        Args:
            z (Variable): shape(B, T), random noise sampled from a standard gaussian disribution.
            condition (Variable, optional): shape(B, F, T), dtype float, the upsampled condition. Defaults to None.

        Returns:
            (z, out_mu, out_log_std)
            z (Variable): shape(B, T), dtype float, transformed noise, it is the synthesized waveform.
            out_mu (Variable): shape(B, T), dtype float, means of the output distributions.
            out_log_std (Variable): shape(B, T), dtype float, log standard deviations of the output distributions.
        """
        for i, flow in enumerate(self.flows):
            theta = flow(z, condition)  # w, mu, log_std [0: T]
            w, mu, log_std = F.split(theta, 3, dim=-1)  # (B, T, 1) for each
            mu = F.squeeze(mu, [-1])  #[0: T]
            log_std = F.squeeze(log_std, [-1])  #[0: T]
            z = z * F.exp(log_std) + mu  #[0: T]

            if i == 0:
                out_mu = mu
                out_log_std = log_std
            else:
                out_mu = out_mu * F.exp(log_std) + mu
                out_log_std += log_std

        return z, out_mu, out_log_std
