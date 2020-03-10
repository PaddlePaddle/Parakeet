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
import paddle.fluid.dygraph as dg


def summary(layer):
    num_params = num_elements = 0
    print("layer summary:")
    for name, param in layer.state_dict().items():
        print("{}|{}|{}".format(name, param.shape, np.prod(param.shape)))
        num_elements += np.prod(param.shape)
        num_params += 1
    print("layer has {} parameters, {} elements.".format(num_params,
                                                         num_elements))


def freeze(layer):
    for param in layer.parameters():
        param.trainable = False


def unfreeze(layer):
    for param in layer.parameters():
        param.trainable = True
