import itertools
import math

import numpy as np
from paddle import fluid
import paddle.fluid.dygraph as dg
import ops
import weight_norm


def get_padding(filter_size, stride, padding_type='same'):
    if padding_type == 'same':
        padding = [(x - y) // 2 for x, y in zip(filter_size, stride)]
    else:
        raise ValueError("Only support same padding")
    return padding


def debug(x, var_name, rank, verbose=False):
    if not verbose and rank != 0:
        return
    dim = len(x.shape)
    if not isinstance(x, np.ndarray):
        x = x.numpy()
    if dim == 1:
        print("Rank {}".format(rank), var_name, "shape {}, value {}".format(x.shape, x))
    elif dim == 2:
        print("Rank {}".format(rank), var_name, "shape {}, value {}".format(x.shape, x[:, :5]))
    elif dim == 3:
        print("Rank {}".format(rank), var_name, "shape {}, value {}".format(x.shape, x[:, :5, 0]))
    else:
        print("Rank", rank, var_name, "shape", x.shape)


def extract_slices(x, audio_starts, audio_length, rank):
    slices = []
    for i in range(x.shape[0]): 
        start = audio_starts.numpy()[i]
        end = start + audio_length
        slice = fluid.layers.slice(
            x, axes=[0, 1], starts=[i, start], ends=[i+1, end])
        slices.append(fluid.layers.squeeze(slice, [0]))

    x = fluid.layers.stack(slices, axis=0)

    return x


class Conditioner(dg.Layer):
    def __init__(self, name_scope, config):
        super(Conditioner, self).__init__(name_scope)
        upsample_factors = config.conditioner.upsample_factors
        filter_sizes = config.conditioner.filter_sizes
        assert np.prod(upsample_factors) == config.fft_window_shift

        self.deconvs = []
        for i, up_scale in enumerate(upsample_factors):
            stride = (up_scale, 1)
            padding = get_padding(filter_sizes[i], stride)
            self.deconvs.append(
                ops.Conv2DTranspose(
                    self.full_name(),
                    num_filters=1,
                    filter_size=filter_sizes[i],
                    padding=padding,
                    stride=stride))

        # Register python list as parameters.
        for i, layer in enumerate(self.deconvs):
            self.add_sublayer("conv_transpose_{}".format(i), layer)
        
    def forward(self, x):
        x = fluid.layers.unsqueeze(x, 1)
        for layer in self.deconvs:
            x = fluid.layers.leaky_relu(layer(x), alpha=0.4)

        return fluid.layers.squeeze(x, [1])


class WaveNetModule(dg.Layer):
    def __init__(self, name_scope, config, rank):
        super(WaveNetModule, self).__init__(name_scope)
        
        self.rank = rank
        self.conditioner = Conditioner(self.full_name(), config)
        self.dilations = list(
            itertools.islice(
                itertools.cycle(config.dilation_block), config.layers))
        self.context_size = sum(self.dilations) + 1
        self.log_scale_min = config.log_scale_min
        self.config = config

        print("dilations", self.dilations)
        print("context_size", self.context_size)

        if config.loss_type == "softmax":
            self.embedding_fc = ops.Embedding(
                self.full_name(),
                num_embeddings=config.num_channels,
                embed_dim=config.residual_channels)
        elif config.loss_type == "mix-gaussian-pdf":
            self.embedding_fc = ops.FC(
                self.full_name(),
                in_features=1,
                size=config.residual_channels,
                num_flatten_dims=2,
                relu=False)
        else:
            raise ValueError(
                "loss_type {} is unsupported!".format(loss_type))

        self.dilated_causal_convs = []
        for dilation in self.dilations:
            self.dilated_causal_convs.append(
                ops.Conv1D_GU(
                    self.full_name(),
                    conditioner_dim=config.mel_bands,
                    in_channels=config.residual_channels,
                    num_filters=config.residual_channels,
                    filter_size=config.kernel_width,
                    dilation=dilation,
                    causal=True
                )
            )

        for i, layer in enumerate(self.dilated_causal_convs):
            self.add_sublayer("dilated_causal_conv_{}".format(i), layer) 

        self.fc1 = ops.FC(
            self.full_name(),
            in_features=config.residual_channels,
            size=config.skip_channels,
            num_flatten_dims=2,
            relu=True,
            act="relu")

        self.fc2 = ops.FC(
            self.full_name(),
            in_features=config.skip_channels,
            size=config.skip_channels,
            num_flatten_dims=2,
            relu=True,
            act="relu")

        if config.loss_type == "softmax":
            self.fc3 = ops.FC(
                self.full_name(),
                in_features=config.skip_channels,
                size=config.num_channels,
                num_flatten_dims=2,
                relu=False)
        elif config.loss_type == "mix-gaussian-pdf":
            self.fc3 = ops.FC(
                self.full_name(),
                in_features=config.skip_channels,
                size=3 * config.num_mixtures,
                num_flatten_dims=2,
                relu=False)
        else:
            raise ValueError(
                "loss_type {} is unsupported!".format(loss_type))

    def sample_softmax(self, mix_parameters):
        batch, length, hidden = mix_parameters.shape
        mix_param_2d = fluid.layers.reshape(mix_parameters,
            [batch * length, hidden])
        mix_param_2d = fluid.layers.softmax(mix_param_2d, axis=-1)

        # quantized: [batch * length]
        quantized = fluid.layers.cast(fluid.layers.sampling_id(mix_param_2d),
            dtype="float32")
        samples = (quantized + 0.5) * (2.0 / self.config.num_channels) - 1.0

        # samples: [batch * length]
        return samples

    def sample_mix_gaussian(self, mix_parameters):
        # mix_parameters reshape from [bs, 13799, 3 * num_mixtures]
        # to [bs * 13799, 3 * num_mixtures].
        batch, length, hidden = mix_parameters.shape
        mix_param_2d = fluid.layers.reshape(mix_parameters,
            [batch * length, hidden])
        K = hidden // 3

        # Unpack the parameters of the mixture of gaussian.
        logits_pi = mix_param_2d[:, 0 : K]
        mu = mix_param_2d[:, K : 2*K]
        log_s = mix_param_2d[:, 2*K : 3*K]
        s = fluid.layers.exp(log_s)

        pi = fluid.layers.softmax(logits_pi, axis=-1)
        comp_samples = fluid.layers.sampling_id(pi)
        
        row_idx = dg.to_variable(np.arange(batch * length))
        comp_samples = fluid.layers.stack([row_idx, comp_samples], axis=-1)

        mu_comp = fluid.layers.gather_nd(mu, comp_samples)
        s_comp = fluid.layers.gather_nd(s, comp_samples) 

        # N(0, 1) Normal Sample.
        u = fluid.layers.gaussian_random(shape=[batch * length])
        samples = mu_comp + u * s_comp
        samples = fluid.layers.clip(samples, min=-1.0, max=1.0)

        return samples

    def softmax_loss(self, targets, mix_parameters):
        # targets: [bs, 13799] -> [bs, 11752]
        # mix_params: [bs, 13799, 3] -> [bs, 11752, 3]
        targets = targets[:, self.context_size:]
        mix_parameters = mix_parameters[:, self.context_size:, :]

        # Quantized audios to integral values with range [0, num_channels)
        num_channels = self.config.num_channels
        targets = fluid.layers.clip(targets, min=-1.0, max=0.99999)
        quantized = fluid.layers.cast(
            (targets + 1.0) / 2.0 * num_channels, dtype="int64")

        # per_sample_loss shape: [bs, 17952, 1]
        per_sample_loss = fluid.layers.softmax_with_cross_entropy(
            logits=mix_parameters, label=fluid.layers.unsqueeze(quantized, 2))
        loss = fluid.layers.reduce_mean(per_sample_loss)
        #debug(loss, "softmax loss", self.rank)

        return loss

    def mixture_density_loss(self, targets, mix_parameters, log_scale_min):
        # targets: [bs, 13799] -> [bs, 11752]
        # mix_params: [bs, 13799, 3] -> [bs, 11752, 3]
        targets = targets[:, self.context_size:]
        mix_parameters = mix_parameters[:, self.context_size:, :]

        # log_s: [bs, 11752, num_mixture]
        logits_pi, mu, log_s = fluid.layers.split(mix_parameters, num_or_sections=3, dim=-1)

        pi = fluid.layers.softmax(logits_pi, axis=-1)
        log_s = fluid.layers.clip(log_s, min=log_scale_min, max=100.0)
        inv_s = fluid.layers.exp(0.0 - log_s)

        # Calculate gaussian loss.
        targets = fluid.layers.unsqueeze(targets, -1)
        targets = fluid.layers.expand(targets, [1, 1, self.config.num_mixtures])
        x_std =  inv_s * (targets - mu)
        exponent = fluid.layers.exp(-0.5 * x_std * x_std)
        # pdf_x: [bs, 11752, 1]
        pdf_x = 1.0 / np.sqrt(2.0 * np.pi) * inv_s * exponent
        pdf_x = pi * pdf_x
        # pdf_x: [bs, 11752]
        pdf_x = fluid.layers.reduce_sum(pdf_x, dim=-1)
        per_sample_loss = 0.0 - fluid.layers.log(pdf_x + 1e-9)

        loss = fluid.layers.reduce_mean(per_sample_loss)

        return loss

    def forward(self, audios, mels, audio_starts, sample=False):
        # audios: [bs, 13800], mels: [bs, full_frame_length, 80]
        # audio_starts: [bs]
        # Build conditioner based on mels.
        full_conditioner = self.conditioner(mels)

        # Slice conditioners.
        audio_length = audios.shape[1]
        conditioner = extract_slices(full_conditioner,
            audio_starts, audio_length, self.rank)
    
        # input_audio, target_audio: [bs, 13799]
        input_audios = audios[:, :-1]
        target_audios = audios[:, 1:]
        # conditioner: [bs, 13799, 80]
        conditioner = conditioner[:, 1:, :]

        loss_type = self.config.loss_type

        # layer_input: [bs, 13799, 128]
        if loss_type == "softmax":
            input_audios = fluid.layers.clip(
                input_audios, min=-1.0, max=0.99999)
            # quantized have values in [0, num_channels)
            quantized = fluid.layers.cast(
                (input_audios + 1.0) / 2.0 * self.config.num_channels,
                dtype="int64")
            layer_input = self.embedding_fc(fluid.layers.unsqueeze(quantized, 2))
        elif loss_type == "mix-gaussian-pdf":
            layer_input = self.embedding_fc(fluid.layers.unsqueeze(input_audios, 2))
        else:
            raise ValueError(
                "loss_type {} is unsupported!".format(loss_type))

        # layer_input: [bs, res_channel, 1, 13799]
        layer_input = fluid.layers.unsqueeze(fluid.layers.transpose(layer_input, perm=[0, 2, 1]), 2)
        # conditioner: [bs, mel_bands, 1, 13799]
        conditioner = fluid.layers.unsqueeze(fluid.layers.transpose(conditioner, perm=[0, 2, 1]), 2)

        # layer_input: [bs, res_channel, 1, 13799]
        # skip: [bs, res_channel, 1, 13799]
        skip = None
        for i, layer in enumerate(self.dilated_causal_convs):
            layer_input, skip = layer(layer_input, skip, conditioner)
            #debug(layer_input, "layer_input_" + str(i), self.rank)
            #debug(skip, "skip_" + str(i), self.rank)

        # Reshape skip to [bs, 13799, res_channel]
        skip = fluid.layers.transpose(fluid.layers.squeeze(skip, [2]), perm=[0, 2, 1])
        #debug(skip, "skip", self.rank)

        # mix_param: [bs, 13799, 3 * num_mixtures]
        mix_parameters = self.fc3(self.fc2(self.fc1(skip)))

        # Sample teacher-forced audio.
        sample_audios = None
        if sample:
            if loss_type == "softmax":
                sample_audios = self.sample_softmax(mix_parameters)
            elif loss_type == "mix-gaussian-pdf":
                sample_audios = self.sample_mix_gaussian(mix_parameters)
            else:
                raise ValueError(
                    "loss_type {} is unsupported!".format(loss_type))
            #debug(sample_audios, "sample_audios", self.rank)

        # Calculate mix-gaussian density loss.
        # padding is all zero.
        # target_audio: [bs, 13799].
        # mix_params: [bs, 13799, 3].
        if loss_type == "softmax":
            loss = self.softmax_loss(target_audios, mix_parameters)
        elif loss_type == "mix-gaussian-pdf":
            loss = self.mixture_density_loss(target_audios,
                mix_parameters, self.log_scale_min)
        else:
            raise ValueError(
                "loss_type {} is unsupported!".format(loss_type))

        #print("Rank {}, loss {}".format(self.rank, loss.numpy()))
 
        return loss, sample_audios

    def synthesize(self, mels):
        self.start_new_sequence()
        print("input mels shape", mels.shape)
        # mels: [bs=1, n_frames, 80]
        # conditioner: [1, n_frames * samples_per_frame, 80]
        # Should I move forward by one sample? No difference
        # Append context frame to mels
        bs, n_frames, mel_bands = mels.shape 
        #num_pad_frames = int(np.ceil(self.context_size / self.config.fft_window_shift))
        #silence = fluid.layers.zeros(shape=[bs, num_pad_frames, mel_bands], dtype="float32")
        #inf_mels = fluid.layers.concat([silence, mels], axis=1)
        #print("padded mels shape", inf_mels.shape)

        #conditioner = self.conditioner(inf_mels)[:, self.context_size:, :]
        conditioner = self.conditioner(mels)
        time_steps = conditioner.shape[1]
        print("Total steps", time_steps)

        loss_type = self.config.loss_type
        audio_samples = []
        current_sample = fluid.layers.zeros(shape=[bs, 1, 1], dtype="float32")
        for i in range(time_steps):
            if i % 100 == 0:
                print("Step", i)

            # convert from real value sample to audio embedding.
            # [bs, 1, 128]
            if loss_type == "softmax":
                current_sample = fluid.layers.clip(
                    current_sample, min=-1.0, max=0.99999)
                # quantized have values in [0, num_channels)
                quantized = fluid.layers.cast(
                    (current_sample + 1.0) / 2.0 * self.config.num_channels,
                    dtype="int64")
                audio_input = self.embedding_fc(quantized)
            elif loss_type == "mix-gaussian-pdf":
                audio_input = self.embedding_fc(current_sample)
            else:
                raise ValueError(
                    "loss_type {} is unsupported!".format(loss_type))

            # [bs, 128, 1, 1]
            audio_input = fluid.layers.unsqueeze(fluid.layers.transpose(audio_input, perm=[0, 2, 1]), 2)
            # [bs, 80]
            cond_input = conditioner[:, i, :]
            # [bs, 80, 1, 1]
            cond_input = fluid.layers.reshape(
                cond_input, cond_input.shape + [1, 1])

            skip = None
            for layer in self.dilated_causal_convs:
                audio_input, skip = layer.add_input(audio_input, skip, cond_input)
            
            # [bs, 1, 128]
            skip = fluid.layers.transpose(fluid.layers.squeeze(skip, [2]), perm=[0, 2, 1])
            # [bs, 1, 3]
            mix_parameters = self.fc3(self.fc2(self.fc1(skip)))
            if loss_type == "softmax":
                sample = self.sample_softmax(mix_parameters)
            elif loss_type == "mix-gaussian-pdf":
                sample = self.sample_mix_gaussian(mix_parameters)
            else:
                raise ValueError(
                    "loss_type {} is unsupported!".format(loss_type))
            audio_samples.append(sample)
            # [bs]
            current_sample = audio_samples[-1]
            # [bs, 1, 1]
            current_sample = fluid.layers.reshape(current_sample,
                current_sample.shape + [1, 1])

        # syn_audio: (num_samples,)
        syn_audio = fluid.layers.concat(audio_samples, axis=0).numpy()

        return syn_audio        

    def start_new_sequence(self):
        for layer in self.sublayers():
            if isinstance(layer, weight_norm.Conv1D):
                layer.start_new_sequence()

    def save(self, iteration):
        utils.save_latest_parameters(self.checkpoint_dir, iteration,
                                     self.wavenet, self.optimizer)
        utils.save_latest_checkpoint(self.checkpoint_dir, iteration)
