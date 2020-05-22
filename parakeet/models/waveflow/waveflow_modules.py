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

import itertools
import numpy as np
import paddle.fluid.dygraph as dg
from paddle import fluid
from parakeet.modules import weight_norm


def get_param_attr(layer_type, filter_size, c_in=1):
    if layer_type == "weight_norm":
        k = np.sqrt(1.0 / (c_in * np.prod(filter_size)))
        weight_init = fluid.initializer.UniformInitializer(low=-k, high=k)
        bias_init = fluid.initializer.UniformInitializer(low=-k, high=k)
    elif layer_type == "common":
        weight_init = fluid.initializer.ConstantInitializer(0.0)
        bias_init = fluid.initializer.ConstantInitializer(0.0)
    else:
        raise TypeError("Unsupported layer type.")

    param_attr = fluid.ParamAttr(initializer=weight_init)
    bias_attr = fluid.ParamAttr(initializer=bias_init)
    return param_attr, bias_attr


def unfold(x, n_group):
    length = x.shape[-1]
    new_shape = x.shape[:-1] + [length // n_group, n_group]
    return fluid.layers.reshape(x, new_shape)


class WaveFlowLoss:
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, model_output):
        z, log_s_list = model_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = fluid.layers.reduce_sum(log_s)
            else:
                log_s_total = log_s_total + fluid.layers.reduce_sum(log_s)

        loss = fluid.layers.reduce_sum(z * z) / (2 * self.sigma * self.sigma) \
            - log_s_total
        loss = loss / np.prod(z.shape)
        const = 0.5 * np.log(2 * np.pi) + np.log(self.sigma)

        return loss + const


class Conditioner(dg.Layer):
    def __init__(self, dtype):
        super(Conditioner, self).__init__()
        upsample_factors = [16, 16]

        self.upsample_conv2d = []
        for s in upsample_factors:
            in_channel = 1
            param_attr, bias_attr = get_param_attr(
                "weight_norm", (3, 2 * s), c_in=in_channel)
            conv_trans2d = weight_norm.Conv2DTranspose(
                num_channels=in_channel,
                num_filters=1,
                filter_size=(3, 2 * s),
                padding=(1, s // 2),
                stride=(1, s),
                param_attr=param_attr,
                bias_attr=bias_attr,
                dtype="float32")
            self.upsample_conv2d.append(conv_trans2d)

        for i, layer in enumerate(self.upsample_conv2d):
            self.add_sublayer("conv2d_transpose_{}".format(i), layer)

    def forward(self, x):
        x = fluid.layers.unsqueeze(x, 1)
        for layer in self.upsample_conv2d:
            in_dtype = x.dtype
            if in_dtype == fluid.core.VarDesc.VarType.FP16:
                x = fluid.layers.cast(x, "float32")
            x = layer(x)
            if in_dtype == fluid.core.VarDesc.VarType.FP16:
                x = fluid.layers.cast(x, "float16")
            x = fluid.layers.leaky_relu(x, alpha=0.4)

        return fluid.layers.squeeze(x, [1])

    def infer(self, x):
        x = fluid.layers.unsqueeze(x, 1)
        for layer in self.upsample_conv2d:
            in_dtype = x.dtype
            if in_dtype == fluid.core.VarDesc.VarType.FP16:
                x = fluid.layers.cast(x, "float32")
            x = layer(x)
            if in_dtype == fluid.core.VarDesc.VarType.FP16:
                x = fluid.layers.cast(x, "float16")
            # Trim conv artifacts.
            time_cutoff = layer._filter_size[1] - layer._stride[1]
            x = fluid.layers.leaky_relu(x[:, :, :, :-time_cutoff], alpha=0.4)

        return fluid.layers.squeeze(x, [1])


class Flow(dg.Layer):
    def __init__(self, config):
        super(Flow, self).__init__()
        self.n_layers = config.n_layers
        self.n_channels = config.n_channels
        self.kernel_h = config.kernel_h
        self.kernel_w = config.kernel_w
        self.dtype = "float16" if config.use_fp16 else "float32"

        # Transform audio: [batch, 1, n_group, time/n_group] 
        # => [batch, n_channels, n_group, time/n_group]
        param_attr, bias_attr = get_param_attr("weight_norm", (1, 1), c_in=1)
        self.start = weight_norm.Conv2D(
            num_channels=1,
            num_filters=self.n_channels,
            filter_size=(1, 1),
            param_attr=param_attr,
            bias_attr=bias_attr,
            dtype=self.dtype)

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        # output shape: [batch, 2, n_group, time/n_group]
        param_attr, bias_attr = get_param_attr(
            "common", (1, 1), c_in=self.n_channels)
        self.end = dg.Conv2D(
            num_channels=self.n_channels,
            num_filters=2,
            filter_size=(1, 1),
            param_attr=param_attr,
            bias_attr=bias_attr,
            dtype=self.dtype)

        # receiptive fileds: (kernel - 1) * sum(dilations) + 1 >= squeeze
        dilation_dict = {
            8: [1, 1, 1, 1, 1, 1, 1, 1],
            16: [1, 1, 1, 1, 1, 1, 1, 1],
            32: [1, 2, 4, 1, 2, 4, 1, 2],
            64: [1, 2, 4, 8, 16, 1, 2, 4],
            128: [1, 2, 4, 8, 16, 32, 64, 1]
        }
        self.dilation_h_list = dilation_dict[config.n_group]

        self.in_layers = []
        self.cond_layers = []
        self.res_skip_layers = []
        for i in range(self.n_layers):
            dilation_h = self.dilation_h_list[i]
            dilation_w = 2**i

            param_attr, bias_attr = get_param_attr(
                "weight_norm", (self.kernel_h, self.kernel_w),
                c_in=self.n_channels)
            in_layer = weight_norm.Conv2D(
                num_channels=self.n_channels,
                num_filters=2 * self.n_channels,
                filter_size=(self.kernel_h, self.kernel_w),
                dilation=(dilation_h, dilation_w),
                param_attr=param_attr,
                bias_attr=bias_attr,
                dtype=self.dtype)
            self.in_layers.append(in_layer)

            param_attr, bias_attr = get_param_attr(
                "weight_norm", (1, 1), c_in=config.mel_bands)
            cond_layer = weight_norm.Conv2D(
                num_channels=config.mel_bands,
                num_filters=2 * self.n_channels,
                filter_size=(1, 1),
                param_attr=param_attr,
                bias_attr=bias_attr,
                dtype=self.dtype)
            self.cond_layers.append(cond_layer)

            if i < self.n_layers - 1:
                res_skip_channels = 2 * self.n_channels
            else:
                res_skip_channels = self.n_channels
            param_attr, bias_attr = get_param_attr(
                "weight_norm", (1, 1), c_in=self.n_channels)
            res_skip_layer = weight_norm.Conv2D(
                num_channels=self.n_channels,
                num_filters=res_skip_channels,
                filter_size=(1, 1),
                param_attr=param_attr,
                bias_attr=bias_attr,
                dtype=self.dtype)
            self.res_skip_layers.append(res_skip_layer)

            self.add_sublayer("in_layer_{}".format(i), in_layer)
            self.add_sublayer("cond_layer_{}".format(i), cond_layer)
            self.add_sublayer("res_skip_layer_{}".format(i), res_skip_layer)

    def forward(self, audio, mel):
        # audio: [bs, 1, n_group, time/group]
        # mel: [bs, mel_bands, n_group, time/n_group]
        audio = self.start(audio)

        for i in range(self.n_layers):
            dilation_h = self.dilation_h_list[i]
            dilation_w = 2**i

            # Pad height dim (n_group): causal convolution
            # Pad width dim (time): dialated non-causal convolution
            pad_top, pad_bottom = (self.kernel_h - 1) * dilation_h, 0
            pad_left = pad_right = int((self.kernel_w - 1) * dilation_w / 2)
            # Using pad2d is a bit faster than using padding in Conv2D directly 
            audio_pad = fluid.layers.pad2d(
                audio, paddings=[pad_top, pad_bottom, pad_left, pad_right])
            hidden = self.in_layers[i](audio_pad)
            cond_hidden = self.cond_layers[i](mel)
            in_acts = hidden + cond_hidden
            out_acts = fluid.layers.tanh(in_acts[:, :self.n_channels, :]) * \
                fluid.layers.sigmoid(in_acts[:, self.n_channels:, :])
            res_skip_acts = self.res_skip_layers[i](out_acts)

            if i < self.n_layers - 1:
                audio += res_skip_acts[:, :self.n_channels, :, :]
                skip_acts = res_skip_acts[:, self.n_channels:, :, :]
            else:
                skip_acts = res_skip_acts

            if i == 0:
                output = skip_acts
            else:
                output += skip_acts

        return self.end(output)

    def infer(self, audio, mel, queues):
        audio = self.start(audio)

        for i in range(self.n_layers):
            dilation_h = self.dilation_h_list[i]
            dilation_w = 2**i

            state_size = dilation_h * (self.kernel_h - 1)
            queue = queues[i]

            if len(queue) == 0:
                for j in range(state_size):
                    queue.append(fluid.layers.zeros_like(audio))

            state = queue[0:state_size]
            state = fluid.layers.concat(state + [audio], axis=2)

            queue.pop(0)
            queue.append(audio)

            # Pad height dim (n_group): causal convolution
            # Pad width dim (time): dialated non-causal convolution
            pad_top, pad_bottom = 0, 0
            pad_left = int((self.kernel_w - 1) * dilation_w / 2)
            pad_right = int((self.kernel_w - 1) * dilation_w / 2)
            state = fluid.layers.pad2d(
                state, paddings=[pad_top, pad_bottom, pad_left, pad_right])
            hidden = self.in_layers[i](state)
            cond_hidden = self.cond_layers[i](mel)
            in_acts = hidden + cond_hidden
            out_acts = fluid.layers.tanh(in_acts[:, :self.n_channels, :]) * \
                      fluid.layers.sigmoid(in_acts[:, self.n_channels:, :])
            res_skip_acts = self.res_skip_layers[i](out_acts)

            if i < self.n_layers - 1:
                audio += res_skip_acts[:, :self.n_channels, :, :]
                skip_acts = res_skip_acts[:, self.n_channels:, :, :]
            else:
                skip_acts = res_skip_acts

            if i == 0:
                output = skip_acts
            else:
                output += skip_acts

        return self.end(output)


class WaveFlowModule(dg.Layer):
    """WaveFlow model implementation.

    Args:
        config (obj): model configuration parameters.

    Returns:
        WaveFlowModule
    """

    def __init__(self, config):
        super(WaveFlowModule, self).__init__()
        self.n_flows = config.n_flows
        self.n_group = config.n_group
        self.n_layers = config.n_layers
        assert self.n_group % 2 == 0
        assert self.n_flows % 2 == 0

        self.dtype = "float16" if config.use_fp16 else "float32"
        self.conditioner = Conditioner(self.dtype)
        self.flows = []
        for i in range(self.n_flows):
            flow = Flow(config)
            self.flows.append(flow)
            self.add_sublayer("flow_{}".format(i), flow)

        self.perms = []
        half = self.n_group // 2
        for i in range(self.n_flows):
            perm = list(range(self.n_group))
            if i < self.n_flows // 2:
                perm = perm[::-1]
            else:
                perm[:half] = reversed(perm[:half])
                perm[half:] = reversed(perm[half:])
            self.perms.append(perm)

    def forward(self, audio, mel):
        """Training forward pass.

        Use a conditioner to upsample mel spectrograms into hidden states.
        These hidden states along with the audio are passed to a stack of Flow
        modules to obtain the final latent variable z and a list of log scaling
        variables, which are then passed to the WaveFlowLoss module to calculate
        the negative log likelihood.

        Args:
            audio (obj): audio samples.
            mel (obj): mel spectrograms.

        Returns:
            z (obj): latent variable.
            log_s_list(list): list of log scaling variables.
        """
        mel = self.conditioner(mel)
        assert mel.shape[2] >= audio.shape[1]
        # Prune out the tail of audio/mel so that time/n_group == 0.
        pruned_len = int(audio.shape[1] // self.n_group * self.n_group)

        if audio.shape[1] > pruned_len:
            audio = audio[:, :pruned_len]
        if mel.shape[2] > pruned_len:
            mel = mel[:, :, :pruned_len]

        # From [bs, mel_bands, time] to [bs, mel_bands, n_group, time/n_group]
        mel = fluid.layers.transpose(unfold(mel, self.n_group), [0, 1, 3, 2])
        # From [bs, time] to [bs, n_group, time/n_group]
        audio = fluid.layers.transpose(unfold(audio, self.n_group), [0, 2, 1])
        # [bs, 1, n_group, time/n_group] 
        audio = fluid.layers.unsqueeze(audio, 1)
        log_s_list = []
        for i in range(self.n_flows):
            inputs = audio[:, :, :-1, :]
            conds = mel[:, :, 1:, :]
            outputs = self.flows[i](inputs, conds)
            log_s = outputs[:, :1, :, :]
            b = outputs[:, 1:, :, :]
            log_s_list.append(log_s)

            audio_0 = audio[:, :, :1, :]
            audio_out = audio[:, :, 1:, :] * fluid.layers.exp(log_s) + b
            audio = fluid.layers.concat([audio_0, audio_out], axis=2)

            # Permute over the height dim.
            audio_slices = [audio[:, :, j, :] for j in self.perms[i]]
            audio = fluid.layers.stack(audio_slices, axis=2)
            mel_slices = [mel[:, :, j, :] for j in self.perms[i]]
            mel = fluid.layers.stack(mel_slices, axis=2)

        z = fluid.layers.squeeze(audio, [1])
        return z, log_s_list

    def synthesize(self, mel, sigma=1.0):
        """Use model to synthesize waveform.

        Use a conditioner to upsample mel spectrograms into hidden states.
        These hidden states along with initial random gaussian latent variable
        are passed to a stack of Flow modules to obtain the audio output.

        Note that we use convolutional queue (https://arxiv.org/abs/1611.09482)
        to cache the intermediate hidden states, which will speed up the
        autoregressive inference over the height dimension. Current
        implementation only supports height dimension (self.n_group) equals
        8 or 16, i.e., where there is no dilation on the height dimension.

        Args:
            mel (obj): mel spectrograms.
            sigma (float, optional): standard deviation of the guassian latent
                variable. Defaults to 1.0.

        Returns:
            audio (obj): synthesized audio.
        """
        if self.dtype == "float16":
            mel = fluid.layers.cast(mel, self.dtype)
        mel = self.conditioner.infer(mel)
        # From [bs, mel_bands, time] to [bs, mel_bands, n_group, time/n_group]
        mel = fluid.layers.transpose(unfold(mel, self.n_group), [0, 1, 3, 2])

        audio = fluid.layers.gaussian_random(
            shape=[mel.shape[0], 1, mel.shape[2], mel.shape[3]], std=sigma)
        if self.dtype == "float16":
            audio = fluid.layers.cast(audio, self.dtype)
        for i in reversed(range(self.n_flows)):
            # Permute over the height dimension.
            audio_slices = [audio[:, :, j, :] for j in self.perms[i]]
            audio = fluid.layers.stack(audio_slices, axis=2)
            mel_slices = [mel[:, :, j, :] for j in self.perms[i]]
            mel = fluid.layers.stack(mel_slices, axis=2)

            audio_list = []
            audio_0 = audio[:, :, 0:1, :]
            audio_list.append(audio_0)
            audio_h = audio_0
            queues = [[] for _ in range(self.n_layers)]

            for h in range(1, self.n_group):
                inputs = audio_h
                conds = mel[:, :, h:(h + 1), :]
                outputs = self.flows[i].infer(inputs, conds, queues)

                log_s = outputs[:, 0:1, :, :]
                b = outputs[:, 1:, :, :]
                audio_h = (audio[:, :, h:(h+1), :] - b) / \
                    fluid.layers.exp(log_s)
                audio_list.append(audio_h)

            audio = fluid.layers.concat(audio_list, axis=2)

        # audio: [bs, n_group, time/n_group]
        audio = fluid.layers.squeeze(audio, [1])
        # audio: [bs, time]
        audio = fluid.layers.reshape(
            fluid.layers.transpose(audio, [0, 2, 1]), [audio.shape[0], -1])
        return audio
