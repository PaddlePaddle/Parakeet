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

from paddle import fluid
import paddle.fluid.initializer as I
import paddle.fluid.dygraph as dg

from parakeet.g2p import en
from parakeet.models.deepvoice3 import Encoder, Decoder, Converter, DeepVoice3, TTSLoss, ConvSpec, WindowRange
from parakeet.utils.layer_tools import summary, freeze


def make_model(config):
    c = config["model"]
    # speaker embedding
    n_speakers = c["n_speakers"]
    speaker_dim = c["speaker_embed_dim"]
    if n_speakers > 1:
        speaker_embed = dg.Embedding(
            (n_speakers, speaker_dim),
            param_attr=I.Normal(scale=c["speaker_embedding_weight_std"]))
    else:
        speaker_embed = None

    # encoder
    h = c["encoder_channels"]
    k = c["kernel_size"]
    encoder_convolutions = (
        ConvSpec(h, k, 1),
        ConvSpec(h, k, 3),
        ConvSpec(h, k, 9),
        ConvSpec(h, k, 27),
        ConvSpec(h, k, 1),
        ConvSpec(h, k, 3),
        ConvSpec(h, k, 9),
        ConvSpec(h, k, 27),
        ConvSpec(h, k, 1),
        ConvSpec(h, k, 3), )
    encoder = Encoder(
        n_vocab=en.n_vocab,
        embed_dim=c["text_embed_dim"],
        n_speakers=n_speakers,
        speaker_dim=speaker_dim,
        embedding_weight_std=c["embedding_weight_std"],
        convolutions=encoder_convolutions,
        dropout=c["dropout"])
    if c["freeze_embedding"]:
        freeze(encoder.embed)

    # decoder
    h = c["decoder_channels"]
    k = c["kernel_size"]
    prenet_convolutions = (ConvSpec(h, k, 1), ConvSpec(h, k, 3))
    attentive_convolutions = (
        ConvSpec(h, k, 1),
        ConvSpec(h, k, 3),
        ConvSpec(h, k, 9),
        ConvSpec(h, k, 27),
        ConvSpec(h, k, 1), )
    attention = [True, False, False, False, True]
    force_monotonic_attention = [True, False, False, False, True]
    window = WindowRange(c["window_backward"], c["window_ahead"])
    decoder = Decoder(
        n_speakers,
        speaker_dim,
        embed_dim=c["text_embed_dim"],
        mel_dim=config["transform"]["n_mels"],
        r=c["outputs_per_step"],
        max_positions=c["max_positions"],
        preattention=prenet_convolutions,
        convolutions=attentive_convolutions,
        attention=attention,
        dropout=c["dropout"],
        use_memory_mask=c["use_memory_mask"],
        force_monotonic_attention=force_monotonic_attention,
        query_position_rate=c["query_position_rate"],
        key_position_rate=c["key_position_rate"],
        window_range=window,
        key_projection=c["key_projection"],
        value_projection=c["value_projection"])
    if not c["trainable_positional_encodings"]:
        freeze(decoder.embed_keys_positions)
        freeze(decoder.embed_query_positions)

    # converter(postnet)
    linear_dim = 1 + config["transform"]["n_fft"] // 2
    h = c["converter_channels"]
    k = c["kernel_size"]
    postnet_convolutions = (
        ConvSpec(h, k, 1),
        ConvSpec(h, k, 3),
        ConvSpec(2 * h, k, 1),
        ConvSpec(2 * h, k, 3), )
    use_decoder_states = c["use_decoder_state_for_postnet_input"]
    converter = Converter(
        n_speakers,
        speaker_dim,
        in_channels=decoder.state_dim
        if use_decoder_states else config["transform"]["n_mels"],
        linear_dim=linear_dim,
        time_upsampling=c["downsample_factor"],
        convolutions=postnet_convolutions,
        dropout=c["dropout"])

    model = DeepVoice3(
        encoder,
        decoder,
        converter,
        speaker_embed,
        use_decoder_states=use_decoder_states)
    return model


def make_criterion(config):
    # =========================loss=========================
    loss_config = config["loss"]
    transform_config = config["transform"]
    model_config = config["model"]

    priority_freq = loss_config["priority_freq"]  # Hz
    sample_rate = transform_config["sample_rate"]
    linear_dim = 1 + transform_config["n_fft"] // 2
    priority_bin = int(priority_freq / (0.5 * sample_rate) * linear_dim)

    criterion = TTSLoss(
        masked_weight=loss_config["masked_loss_weight"],
        priority_bin=priority_bin,
        priority_weight=loss_config["priority_freq_weight"],
        binary_divergence_weight=loss_config["binary_divergence_weight"],
        guided_attention_sigma=loss_config["guided_attention_sigma"],
        downsample_factor=model_config["downsample_factor"],
        r=model_config["outputs_per_step"])
    return criterion


def make_optimizer(model, config):
    # =========================lr_scheduler=========================
    lr_config = config["lr_scheduler"]
    warmup_steps = lr_config["warmup_steps"]
    peak_learning_rate = lr_config["peak_learning_rate"]
    lr_scheduler = dg.NoamDecay(1 / (warmup_steps * (peak_learning_rate)**2),
                                warmup_steps)

    # =========================optimizer=========================
    optim_config = config["optimizer"]
    optim = fluid.optimizer.Adam(
        lr_scheduler,
        beta1=optim_config["beta1"],
        beta2=optim_config["beta2"],
        epsilon=optim_config["epsilon"],
        parameter_list=model.parameters(),
        grad_clip=fluid.clip.GradientClipByGlobalNorm(0.1))
    return optim
