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

from paddle import fluid
from paddle.fluid.core import ops


@fluid.framework.dygraph_only
def conv2d(input,
           weight,
           stride=(1, 1),
           padding=((0, 0), (0, 0)),
           dilation=(1, 1),
           groups=1,
           use_cudnn=True,
           data_format="NCHW"):
    padding = tuple(pad for pad_dim in padding for pad in pad_dim)

    attrs = ('strides', stride, 'paddings', padding, 'dilations', dilation,
             'groups', groups, 'use_cudnn', use_cudnn, 'use_mkldnn', False,
             'fuse_relu_before_depthwise_conv', False, "padding_algorithm",
             "EXPLICIT", "data_format", data_format)

    out = ops.conv2d(input, weight, *attrs)
    return out
