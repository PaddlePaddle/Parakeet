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

import paddle
import torch
from timer import timer
from parallel_wavegan.layers import upsample, residual_block
from parallel_wavegan.models import parallel_wavegan as pwgan
from parakeet.utils.layer_tools import summary
from parakeet.utils.profile import synchronize

from parakeet.models.parallel_wavegan import ConvInUpsampleNet, ResidualBlock
from parakeet.models.parallel_wavegan import PWGGenerator, PWGDiscriminator, ResidualPWGDiscriminator

paddle.set_device("gpu:0")
device = torch.device("cuda:0")


def test_convin_upsample_net():
    net = ConvInUpsampleNet(
        [4, 4, 4, 4],
        "LeakyReLU", {"negative_slope": 0.2},
        freq_axis_kernel_size=3,
        aux_context_window=0)
    net2 = upsample.ConvInUpsampleNetwork(
        [4, 4, 4, 4],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        freq_axis_kernel_size=3,
        aux_context_window=0).to(device)
    summary(net)
    for k, v in net2.named_parameters():
        print(k, v.shape)
        net.state_dict()[k].set_value(v.data.cpu().numpy())

    c = paddle.randn([4, 80, 180])
    synchronize()
    with timer(unit='s') as t:
        out = net(c)
        synchronize()
        print(f"paddle conv_in_upsample_net forward takes {t.elapse}s.")

    with timer(unit='s') as t:
        out.sum().backward()
        synchronize()
        print(f"paddle conv_in_upsample_net backward takes {t.elapse}s.")

    c_torch = torch.as_tensor(c.numpy()).to(device)
    torch.cuda.synchronize()
    with timer(unit='s') as t:
        out2 = net2(c_torch)
        print(f"torch conv_in_upsample_net forward takes {t.elapse}s.")

    with timer(unit='s') as t:
        out2.sum().backward()
        print(f"torch conv_in_upsample_net backward takes {t.elapse}s.")

    print(out.numpy()[0])
    print(out2.data.cpu().numpy()[0])


def test_residual_block():
    net = ResidualBlock(dilation=9)
    net2 = residual_block.ResidualBlock(dilation=9)
    summary(net)
    summary(net2)
    for k, v in net2.named_parameters():
        net.state_dict()[k].set_value(v.data.cpu().numpy())

    x = paddle.randn([4, 64, 180])
    c = paddle.randn([4, 80, 180])
    res, skip = net(x, c)
    res2, skip2 = net2(torch.as_tensor(x.numpy()), torch.as_tensor(c.numpy()))
    print(res.numpy()[0])
    print(res2.data.cpu().numpy()[0])
    print(skip.numpy()[0])
    print(skip2.data.cpu().numpy()[0])


def test_pwg_generator():
    net = PWGGenerator(
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2})
    net2 = pwgan.ParallelWaveGANGenerator(upsample_params={
        "upsample_scales": [4, 4, 4, 4],
        "nonlinear_activation": "LeakyReLU",
        "nonlinear_activation_params": {
            "negative_slope": 0.2
        }
    }).to(device)
    summary(net)
    summary(net2)
    for k, v in net2.named_parameters():
        p = net.state_dict()[k]
        if k.endswith("_g"):
            p.set_value(v.data.cpu().numpy().reshape([-1]))
        else:
            p.set_value(v.data.cpu().numpy())
    x = paddle.randn([4, 1, 180 * 256])
    c = paddle.randn([4, 80, 180 + 4])

    synchronize()
    with timer(unit='s') as t:
        out = net(x, c)
        synchronize()
        print(f"paddle generator forward takes {t.elapse}s.")

    synchronize()
    with timer(unit='s') as t:
        out.sum().backward()
        synchronize()
        print(f"paddle generator backward takes {t.elapse}s.")

    x_torch = torch.as_tensor(x.numpy()).to(device)
    c_torch = torch.as_tensor(c.numpy()).to(device)

    torch.cuda.synchronize()
    with timer(unit='s') as t:
        out2 = net2(x_torch, c_torch)
        torch.cuda.synchronize()
        print(f"torch generator forward takes {t.elapse}s.")

    torch.cuda.synchronize()
    with timer(unit='s') as t:
        out2.sum().backward()
        torch.cuda.synchronize()
        print(f"torch generator backward takes {t.elapse}s.")

    print(out.numpy()[0])
    print(out2.data.cpu().numpy()[0])
    # print(out.shape)


def test_pwg_discriminator():
    net = PWGDiscriminator()
    net2 = pwgan.ParallelWaveGANDiscriminator()
    summary(net)
    summary(net2)
    for k, v in net2.named_parameters():
        p = net.state_dict()[k]
        if k.endswith("_g"):
            p.set_value(v.data.cpu().numpy().reshape([-1]))
        else:
            p.set_value(v.data.cpu().numpy())
    x = paddle.randn([4, 1, 180 * 256])
    y = net(x)
    y2 = net2(torch.as_tensor(x.numpy()))
    print(y.numpy()[0])
    print(y2.data.cpu().numpy()[0])
    print(y.shape)


def test_residual_pwg_discriminator():
    net = ResidualPWGDiscriminator()
    net2 = pwgan.ResidualParallelWaveGANDiscriminator()
    summary(net)
    summary(net2)
    for k, v in net2.named_parameters():
        p = net.state_dict()[k]
        if k.endswith("_g"):
            p.set_value(v.data.cpu().numpy().reshape([-1]))
        else:
            p.set_value(v.data.cpu().numpy())
    x = paddle.randn([4, 1, 180 * 256])
    y = net(x)
    y2 = net2(torch.as_tensor(x.numpy()))
    print(y.numpy()[0])
    print(y2.data.cpu().numpy()[0])
    print(y.shape)
