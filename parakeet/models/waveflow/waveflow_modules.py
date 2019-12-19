import itertools

import numpy as np
import paddle.fluid.dygraph as dg
from paddle import fluid
from parakeet.modules import conv, modules, weight_norm


def set_param_attr(layer, c_in=1):
    if isinstance(layer, (weight_norm.Conv2DTranspose, weight_norm.Conv2D)):
        k = np.sqrt(1.0 / (c_in * np.prod(layer._filter_size)))
        weight_init = fluid.initializer.UniformInitializer(low=-k, high=k)
        bias_init = fluid.initializer.UniformInitializer(low=-k, high=k)
    elif isinstance(layer, dg.Conv2D):
        weight_init = fluid.initializer.ConstantInitializer(0.0)
        bias_init = fluid.initializer.ConstantInitializer(0.0)
    else:
        raise TypeError("Unsupported layer type.")

    layer._param_attr = fluid.ParamAttr(initializer=weight_init)
    layer._bias_attr = fluid.ParamAttr(initializer=bias_init)


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
    def __init__(self, name_scope):
        super(Conditioner, self).__init__(name_scope)
        upsample_factors = [16, 16]
        
        self.upsample_conv2d = []
        for s in upsample_factors:
            in_channel = 1
            conv_trans2d = modules.Conv2DTranspose(
                self.full_name(),
                num_filters=1,
                filter_size=(3, 2 * s),
                padding=(1, s // 2),
                stride=(1, s))
            set_param_attr(conv_trans2d, c_in=in_channel) 
            self.upsample_conv2d.append(conv_trans2d)

        for i, layer in enumerate(self.upsample_conv2d):
            self.add_sublayer("conv2d_transpose_{}".format(i), layer)

    def forward(self, x):
        x = fluid.layers.unsqueeze(x, 1)
        for layer in self.upsample_conv2d:
            x = fluid.layers.leaky_relu(layer(x), alpha=0.4)

        return fluid.layers.squeeze(x, [1])

    def infer(self, x):
        x = fluid.layers.unsqueeze(x, 1)
        for layer in self.upsample_conv2d:
            x = layer(x)
            # Trim conv artifacts.
            time_cutoff = layer._filter_size[1] - layer._stride[1]
            x = fluid.layers.leaky_relu(x[:, :, :, :-time_cutoff], alpha=0.4)

        return fluid.layers.squeeze(x, [1])


class Flow(dg.Layer):
    def __init__(self, name_scope, config): 
        super(Flow, self).__init__(name_scope)
        self.n_layers = config.n_layers
        self.n_channels = config.n_channels
        self.kernel_h = config.kernel_h
        self.kernel_w = config.kernel_w

        # Transform audio: [batch, 1, n_group, time/n_group] 
        # => [batch, n_channels, n_group, time/n_group]
        self.start = weight_norm.Conv2D(
            self.full_name(),
            num_filters=self.n_channels,
            filter_size=(1, 1))
        set_param_attr(self.start, c_in=1)

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        # output shape: [batch, 2, n_group, time/n_group]
        self.end = dg.Conv2D(
            self.full_name(),
            num_filters=2,
            filter_size=(1, 1))
        set_param_attr(self.end)

        # receiptive fileds: (kernel - 1) * sum(dilations) + 1 >= squeeze
        dilation_dict = {8:   [1, 1, 1, 1, 1, 1, 1, 1],
                         16:  [1, 1, 1, 1, 1, 1, 1, 1],
                         32:  [1, 2, 4, 1, 2, 4, 1, 2],
                         64:  [1, 2, 4, 8, 16, 1, 2, 4],
                         128: [1, 2, 4, 8, 16, 32, 64, 1]}
        self.dilation_h_list = dilation_dict[config.n_group]

        self.in_layers = []
        self.cond_layers = []
        self.res_skip_layers = []
        for i in range(self.n_layers):
            dilation_h = self.dilation_h_list[i]
            dilation_w = 2 ** i

            in_layer = weight_norm.Conv2D(
                self.full_name(),
                num_filters=2 * self.n_channels,
                filter_size=(self.kernel_h, self.kernel_w),
                dilation=(dilation_h, dilation_w))
            set_param_attr(in_layer, c_in=self.n_channels)
            self.in_layers.append(in_layer)

            cond_layer = weight_norm.Conv2D(
                self.full_name(),
                num_filters=2 * self.n_channels,
                filter_size=(1, 1))
            set_param_attr(cond_layer, c_in=config.mel_bands)
            self.cond_layers.append(cond_layer)

            if i < self.n_layers - 1:
                res_skip_channels = 2 * self.n_channels
            else:
                res_skip_channels = self.n_channels
            res_skip_layer = weight_norm.Conv2D(
                self.full_name(),
                num_filters=res_skip_channels,
                filter_size=(1, 1))
            set_param_attr(res_skip_layer, c_in=self.n_channels)
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
            dilation_w = 2 ** i

            # Pad height dim (n_group): causal convolution
            # Pad width dim (time): dialated non-causal convolution
            pad_top, pad_bottom = (self.kernel_h - 1) * dilation_h, 0
            pad_left = pad_right = int((self.kernel_w-1) * dilation_w / 2)
            audio_pad = fluid.layers.pad2d(audio, 
                paddings=[pad_top, pad_bottom, pad_left, pad_right])

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
            dilation_w = 2 ** i

            state_size = dilation_h * (self.kernel_h - 1)
            queue = queues[i]

            if len(queue) == 0:
                for j in range(state_size):
                    queue.append(fluid.layers.zeros_like(audio))

            state = queue[0:state_size]
            state = fluid.layers.concat([*state, audio], axis=2)

            queue.pop(0)
            queue.append(audio)

            # Pad height dim (n_group): causal convolution
            # Pad width dim (time): dialated non-causal convolution
            pad_top, pad_bottom = 0, 0
            pad_left = int((self.kernel_w-1) * dilation_w / 2)
            pad_right = int((self.kernel_w-1) * dilation_w / 2)
            state = fluid.layers.pad2d(state,
                paddings=[pad_top, pad_bottom, pad_left, pad_right])

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
    def __init__(self, name_scope, config):
        super(WaveFlowModule, self).__init__(name_scope)
        self.n_flows = config.n_flows
        self.n_group = config.n_group
        self.n_layers = config.n_layers
        assert self.n_group % 2 == 0
        assert self.n_flows % 2 == 0

        self.conditioner = Conditioner(self.full_name())
        self.flows = []
        for i in range(self.n_flows):
            flow = Flow(self.full_name(), config)
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
        mel = self.conditioner(mel)
        assert mel.shape[2] >= audio.shape[1]
        # Prune out the tail of audio/mel so that time/n_group == 0.
        pruned_len = audio.shape[1] // self.n_group * self.n_group

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
        mel = self.conditioner.infer(mel)
        # From [bs, mel_bands, time] to [bs, mel_bands, n_group, time/n_group]
        mel = fluid.layers.transpose(unfold(mel, self.n_group), [0, 1, 3, 2])

        audio = fluid.layers.gaussian_random(
            shape=[mel.shape[0], 1, mel.shape[2], mel.shape[3]], std=sigma)

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
                conds = mel[:, :, h:(h+1), :]
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
