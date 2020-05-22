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
from paddle import fluid
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as F

from parakeet.modules import customized as L


def norm(param, dim, power):
    powered = F.pow(param, power)
    in_dtype = powered.dtype
    if in_dtype == fluid.core.VarDesc.VarType.FP16:
        powered = F.cast(powered, "float32")
    powered_norm = F.reduce_sum(powered, dim=dim, keep_dim=False)
    norm_ = F.pow(powered_norm, 1. / power)
    if in_dtype == fluid.core.VarDesc.VarType.FP16:
        norm_ = F.cast(norm_, "float16")
    return norm_


def norm_except(param, dim, power):
    """Computes the norm over all dimensions except dim.
    It differs from pytorch implementation that it does not keep dim.
    This difference is related with the broadcast mechanism in paddle.
    Read elementeise_mul for more.
    """
    shape = param.shape
    ndim = len(shape)

    if dim is None:
        return norm(param, dim, power)
    elif dim == 0:
        param_matrix = F.reshape(param, (shape[0], np.prod(shape[1:])))
        return norm(param_matrix, dim=1, power=power)
    elif dim == -1 or dim == ndim - 1:
        param_matrix = F.reshape(param, (np.prod(shape[:-1]), shape[-1]))
        return norm(param_matrix, dim=0, power=power)
    else:
        perm = list(range(ndim))
        perm[0] = dim
        perm[dim] = 0
        transposed_param = F.transpose(param, perm)
        return norm_except(transposed_param, dim=0, power=power)


def compute_l2_normalized_weight(v, g, dim):
    shape = v.shape
    ndim = len(shape)

    if dim is None:
        v_normalized = v / (F.reduce_sum(F.square(v)) + 1e-12)
    elif dim == 0:
        param_matrix = F.reshape(v, (shape[0], np.prod(shape[1:])))
        v_normalized = F.l2_normalize(param_matrix, axis=1)
    elif dim == -1 or dim == ndim - 1:
        param_matrix = F.reshape(v, (np.prod(shape[:-1]), shape[-1]))
        v_normalized = F.l2_normalize(param_matrix, axis=0)
    else:
        perm = list(range(ndim))
        perm[0] = dim
        perm[dim] = 0
        transposed_param = F.transpose(v, perm)
        param_matrix = F.reshape(
            transposed_param,
            (transposed_param.shape[0], np.prod(transposed_param.shape[1:])))
        v_normalized = F.l2_normalize(param_matrix, axis=1)
        v_normalized = F.transpose(v_normalized, perm)
    v_normalized = F.reshape(v_normalized, shape)
    weight = F.elementwise_mul(v_normalized, g, axis=dim)
    return weight


def compute_weight(v, g, dim, power):
    assert len(g.shape) == 1, "magnitude should be a vector"
    if power == 2:
        in_dtype = v.dtype
        if in_dtype == fluid.core.VarDesc.VarType.FP16:
            v = F.cast(v, "float32")
            g = F.cast(g, "float32")
        weight = compute_l2_normalized_weight(v, g, dim)
        if in_dtype == fluid.core.VarDesc.VarType.FP16:
            weight = F.cast(weight, "float16")
        return weight
    else:
        v_normalized = F.elementwise_div(
            v, (norm_except(v, dim, power) + 1e-12), axis=dim)
        weight = F.elementwise_mul(v_normalized, g, axis=dim)
        return weight


class WeightNormWrapper(dg.Layer):
    def __init__(self, layer, param_name="weight", dim=0, power=2):
        super(WeightNormWrapper, self).__init__()

        self.param_name = param_name
        self.dim = dim
        self.power = power
        self.layer = layer

        w_v = param_name + "_v"
        w_g = param_name + "_g"

        # we could also use numpy to compute this, after all, it is run only once
        # at initialization.
        original_weight = getattr(layer, param_name)
        self.add_parameter(
            w_v,
            self.create_parameter(
                shape=original_weight.shape, dtype=original_weight.dtype))
        with dg.no_grad():
            F.assign(original_weight, getattr(self, w_v))
        delattr(layer, param_name)
        temp = norm_except(getattr(self, w_v), self.dim, self.power)
        self.add_parameter(
            w_g, self.create_parameter(
                shape=temp.shape, dtype=temp.dtype))
        with dg.no_grad():
            F.assign(temp, getattr(self, w_g))

        # also set this when setting up
        setattr(self.layer, self.param_name,
                compute_weight(
                    getattr(self, w_v),
                    getattr(self, w_g), self.dim, self.power))

        self.weigth_norm_applied = True

    # hook to compute weight with v & g
    def hook(self):
        w_v = self.param_name + "_v"
        w_g = self.param_name + "_g"
        setattr(self.layer, self.param_name,
                compute_weight(
                    getattr(self, w_v),
                    getattr(self, w_g), self.dim, self.power))

    def remove_weight_norm(self):
        self.hook()
        self.weigth_norm_applied = False

    def forward(self, *args, **kwargs):
        if self.weigth_norm_applied == True:
            self.hook()
        return self.layer(*args, **kwargs)

    def __getattr__(self, key):
        """
        this is used for attr forwarding.
        """
        if key in self._parameters:
            return self._parameters[key]
        elif key in self._sub_layers:
            return self._sub_layers[key]
        elif key is "layer":
            return self._sub_layers["layer"]
        else:
            return getattr(
                object.__getattribute__(self, "_sub_layers")["layer"], key)


def Linear(input_dim,
           output_dim,
           param_attr=None,
           bias_attr=None,
           act=None,
           dtype="float32"):
    # a weight norm applied linear layer.
    lin = dg.Linear(input_dim, output_dim, param_attr, bias_attr, act, dtype)
    lin = WeightNormWrapper(lin, dim=1)
    return lin


def Conv1D(num_channels,
           num_filters,
           filter_size,
           stride=1,
           padding=0,
           dilation=1,
           groups=1,
           param_attr=None,
           bias_attr=None,
           use_cudnn=True,
           act=None,
           dtype='float32'):
    conv = L.Conv1D(num_channels, num_filters, filter_size, stride, padding,
                    dilation, groups, param_attr, bias_attr, use_cudnn, act,
                    dtype)
    conv = WeightNormWrapper(conv, dim=0)
    return conv


def Conv1DTranspose(num_channels,
                    num_filters,
                    filter_size,
                    padding=0,
                    stride=1,
                    dilation=1,
                    groups=1,
                    param_attr=None,
                    bias_attr=None,
                    use_cudnn=True,
                    act=None,
                    dtype='float32'):
    conv = L.Conv1DTranspose(num_channels, num_filters, filter_size, padding,
                             stride, dilation, groups, param_attr, bias_attr,
                             use_cudnn, act, dtype)
    conv = WeightNormWrapper(conv, dim=0)
    return conv


def Conv1DCell(num_channels,
               num_filters,
               filter_size,
               dilation=1,
               causal=False,
               groups=1,
               param_attr=None,
               bias_attr=None,
               use_cudnn=True,
               act=None,
               dtype='float32'):
    conv = L.Conv1DCell(num_channels, num_filters, filter_size, dilation,
                        causal, groups, param_attr, bias_attr, use_cudnn, act,
                        dtype)
    conv = WeightNormWrapper(conv, dim=0)
    return conv


def Conv2D(num_channels,
           num_filters,
           filter_size,
           stride=1,
           padding=0,
           dilation=1,
           groups=1,
           param_attr=None,
           bias_attr=None,
           use_cudnn=True,
           act=None,
           dtype='float32'):
    # a conv2d layer with weight norm wrapper
    conv = dg.Conv2D(num_channels, num_filters, filter_size, stride, padding,
                     dilation, groups, param_attr, bias_attr, use_cudnn, act,
                     dtype)
    conv = WeightNormWrapper(conv, dim=0)
    return conv


def Conv2DTranspose(num_channels,
                    num_filters,
                    filter_size,
                    output_size=None,
                    padding=0,
                    stride=1,
                    dilation=1,
                    groups=1,
                    param_attr=None,
                    bias_attr=None,
                    use_cudnn=True,
                    act=None,
                    dtype='float32'):
    # a conv2d transpose layer with weight norm wrapper.
    conv = dg.Conv2DTranspose(num_channels, num_filters, filter_size,
                              output_size, padding, stride, dilation, groups,
                              param_attr, bias_attr, use_cudnn, act, dtype)
    conv = WeightNormWrapper(conv, dim=0)
    return conv
