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
"""Fastspeech2 related modules for paddle"""
import logging
import numpy as np

from typing import Dict
from typing import Sequence
from typing import Tuple

from typeguard import check_argument_types

import paddle
from paddle import nn

from parakeet.modules.fastspeech2_predictor.duration_predictor import DurationPredictor
from parakeet.modules.fastspeech2_predictor.duration_predictor import DurationPredictorLoss
from parakeet.modules.fastspeech2_predictor.variance_predictor import VariancePredictor
from parakeet.modules.fastspeech2_predictor.length_regulator import LengthRegulator
from parakeet.modules.fastspeech2_predictor.postnet import Postnet
from parakeet.modules.nets_utils import make_non_pad_mask
from parakeet.modules.nets_utils import make_pad_mask
from parakeet.modules.fastspeech2_transformer.embedding import PositionalEncoding
from parakeet.modules.fastspeech2_transformer.embedding import ScaledPositionalEncoding
from parakeet.modules.fastspeech2_transformer.encoder import Encoder as TransformerEncoder


class FastSpeech2(nn.Layer):
    """FastSpeech2 module.

    This is a module of FastSpeech2 described in `FastSpeech 2: Fast and
    High-Quality End-to-End Text to Speech`_. Instead of quantized pitch and
    energy, we use token-averaged value introduced in `FastPitch: Parallel
    Text-to-speech with Pitch Prediction`_.

    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558
    .. _`FastPitch: Parallel Text-to-speech with Pitch Prediction`:
        https://arxiv.org/abs/2006.06873

    """

    def __init__(
            self,
            # network structure related
            idim: int,
            odim: int,
            adim: int=384,
            aheads: int=4,
            elayers: int=6,
            eunits: int=1536,
            dlayers: int=6,
            dunits: int=1536,
            postnet_layers: int=5,
            postnet_chans: int=512,
            postnet_filts: int=5,
            positionwise_layer_type: str="conv1d",
            positionwise_conv_kernel_size: int=1,
            use_scaled_pos_enc: bool=True,
            use_batch_norm: bool=True,
            encoder_normalize_before: bool=True,
            decoder_normalize_before: bool=True,
            encoder_concat_after: bool=False,
            decoder_concat_after: bool=False,
            reduction_factor: int=1,
            encoder_type: str="transformer",
            decoder_type: str="transformer",
            # duration predictor
            duration_predictor_layers: int=2,
            duration_predictor_chans: int=384,
            duration_predictor_kernel_size: int=3,
            # energy predictor
            energy_predictor_layers: int=2,
            energy_predictor_chans: int=384,
            energy_predictor_kernel_size: int=3,
            energy_predictor_dropout: float=0.5,
            energy_embed_kernel_size: int=9,
            energy_embed_dropout: float=0.5,
            stop_gradient_from_energy_predictor: bool=False,
            # pitch predictor
            pitch_predictor_layers: int=2,
            pitch_predictor_chans: int=384,
            pitch_predictor_kernel_size: int=3,
            pitch_predictor_dropout: float=0.5,
            pitch_embed_kernel_size: int=9,
            pitch_embed_dropout: float=0.5,
            stop_gradient_from_pitch_predictor: bool=False,
            # training related
            transformer_enc_dropout_rate: float=0.1,
            transformer_enc_positional_dropout_rate: float=0.1,
            transformer_enc_attn_dropout_rate: float=0.1,
            transformer_dec_dropout_rate: float=0.1,
            transformer_dec_positional_dropout_rate: float=0.1,
            transformer_dec_attn_dropout_rate: float=0.1,
            duration_predictor_dropout_rate: float=0.1,
            postnet_dropout_rate: float=0.5,
            init_type: str="xavier_uniform",
            init_enc_alpha: float=1.0,
            init_dec_alpha: float=1.0,
            use_masking: bool=False,
            use_weighted_masking: bool=False, ):
        """Initialize FastSpeech2 module."""
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.reduction_factor = reduction_factor
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor
        self.use_scaled_pos_enc = use_scaled_pos_enc

        # use idx 0 as padding idx
        self.padding_idx = 0

        # get positional encoding class
        pos_enc_class = (ScaledPositionalEncoding
                         if self.use_scaled_pos_enc else PositionalEncoding)

        # define encoder
        encoder_input_layer = nn.Embedding(
            num_embeddings=idim,
            embedding_dim=adim,
            padding_idx=self.padding_idx)
        print("encoder_type:", encoder_type)
        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                idim=idim,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=encoder_input_layer,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size, )
        else:
            print("encoder_type:", encoder_type)
            raise ValueError(f"{encoder_type} is not supported.")

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=adim,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate, )

        # define pitch predictor
        self.pitch_predictor = VariancePredictor(
            idim=adim,
            n_layers=pitch_predictor_layers,
            n_chans=pitch_predictor_chans,
            kernel_size=pitch_predictor_kernel_size,
            dropout_rate=pitch_predictor_dropout, )
        #  We use continuous pitch + FastPitch style avg
        self.pitch_embed = nn.Sequential(
            nn.Conv1D(
                in_channels=1,
                out_channels=adim,
                kernel_size=pitch_embed_kernel_size,
                padding=(pitch_embed_kernel_size - 1) // 2, ),
            nn.Dropout(pitch_embed_dropout), )

        # define energy predictor
        self.energy_predictor = VariancePredictor(
            idim=adim,
            n_layers=energy_predictor_layers,
            n_chans=energy_predictor_chans,
            kernel_size=energy_predictor_kernel_size,
            dropout_rate=energy_predictor_dropout, )
        # We use continuous enegy + FastPitch style avg
        self.energy_embed = nn.Sequential(
            nn.Conv1D(
                in_channels=1,
                out_channels=adim,
                kernel_size=energy_embed_kernel_size,
                padding=(energy_embed_kernel_size - 1) // 2, ),
            nn.Dropout(energy_embed_dropout), )

        # define length regulator
        self.length_regulator = LengthRegulator()

        # define decoder
        # NOTE: we use encoder as decoder
        # because fastspeech's decoder is the same as encoder
        if decoder_type == "transformer":
            self.decoder = TransformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=dunits,
                num_blocks=dlayers,
                input_layer=None,
                dropout_rate=transformer_dec_dropout_rate,
                positional_dropout_rate=transformer_dec_positional_dropout_rate,
                attention_dropout_rate=transformer_dec_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=decoder_normalize_before,
                concat_after=decoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size, )
        else:
            raise ValueError(f"{decoder_type} is not supported.")

        # define final projection
        self.feat_out = nn.Linear(adim, odim * reduction_factor)

        # define postnet
        self.postnet = (None if postnet_layers == 0 else Postnet(
            idim=idim,
            odim=odim,
            n_layers=postnet_layers,
            n_chans=postnet_chans,
            n_filts=postnet_filts,
            use_batch_norm=use_batch_norm,
            dropout_rate=postnet_dropout_rate, ))

        # define criterions
        self.criterion = FastSpeech2Loss(
            use_masking=use_masking, use_weighted_masking=use_weighted_masking)

    def forward(
            self,
            text: paddle.Tensor,
            text_lengths: paddle.Tensor,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            durations: paddle.Tensor,
            durations_lengths: paddle.Tensor,
            pitch: paddle.Tensor,
            pitch_lengths: paddle.Tensor,
            energy: paddle.Tensor,
            energy_lengths: paddle.Tensor, ) -> Tuple[paddle.Tensor, Dict[
                str, paddle.Tensor], paddle.Tensor]:
        # """Calculate forward propagation.

        # Args:
        #     text (LongTensor): Batch of padded token ids (B, Tmax).
        #     text_lengths (LongTensor): Batch of lengths of each input (B,).
        #     speech (Tensor): Batch of padded target features (B, Lmax, odim).
        #     speech_lengths (LongTensor): Batch of the lengths of each target (B,).
        #     durations (LongTensor): Batch of padded durations (B, Tmax + 1).
        #     durations_lengths (LongTensor): Batch of duration lengths (B, Tmax + 1).
        #     pitch (Tensor): Batch of padded token-averaged pitch (B, Tmax + 1, 1).
        #     pitch_lengths (LongTensor): Batch of pitch lengths (B, Tmax + 1).
        #     energy (Tensor): Batch of padded token-averaged energy (B, Tmax + 1, 1).
        #     energy_lengths (LongTensor): Batch of energy lengths (B, Tmax + 1).
        # Returns:
        #     Tensor: Loss scalar value.
        #     Dict: Statistics to be monitored.
        #     Tensor: Weight value.

        # """
        text = text[:, :text_lengths.max()]  # for data-parallel
        speech = speech[:, :speech_lengths.max()]  # for data-parallel
        durations = durations[:, :durations_lengths.max()]  # for data-parallel
        pitch = pitch[:, :pitch_lengths.max()]  # for data-parallel
        energy = energy[:, :energy_lengths.max()]  # for data-parallel

        batch_size = text.shape[0]

        # Add eos at the last of sequence
        # xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        print("xs.shape in fastspeech2.py before:", text.shape, text)
        xs = np.pad(text.numpy(),
                    pad_width=((0, 0), (0, 1)),
                    mode="constant",
                    constant_values=self.padding_idx)
        xs = paddle.to_tensor(xs)
        print("xs.shape in fastspeech2.py end:", xs.shape, xs)
        # my_pad = nn.Pad1D(padding=[0, 1], mode="constant", value=self.padding_idx)
        # xs = my_pad(text)
        # 是否会数组越界？ xs 是否能取到 l -> 可以，因为上一步补充了一个 padding_idx，又变成了 eos
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos
        ilens = text_lengths + 1

        ys, ds, ps, es = speech, durations, pitch, energy
        olens = speech_lengths

        # forward propagation
        before_outs, after_outs, d_outs, p_outs, e_outs = self._forward(
            xs, ilens, ys, olens, ds, ps, es, is_inference=False)
        print("d_outs in paddle:", d_outs)
        print("p_outs in paddle:", p_outs)
        print("e_outs in paddle:", e_outs)

        # modify mod part of groundtruth
        if self.reduction_factor > 1:
            # 需要改
            olens = paddle.to_tensor([
                olen - olen % self.reduction_factor for olen in olens.numpy()
            ])
            max_olen = max(olens)
            ys = ys[:, :max_olen]

        # calculate loss
        if self.postnet is None:
            after_outs = None

        # calculate loss
        l1_loss, duration_loss, pitch_loss, energy_loss = self.criterion(
            after_outs=after_outs,
            before_outs=before_outs,
            d_outs=d_outs,
            p_outs=p_outs,
            e_outs=e_outs,
            ys=ys,
            ds=ds,
            ps=ps,
            es=es,
            ilens=ilens,
            olens=olens, )
        loss = l1_loss + duration_loss + pitch_loss + energy_loss

        stats = dict(
            l1_loss=l1_loss.item(),
            duration_loss=duration_loss.item(),
            pitch_loss=pitch_loss.item(),
            energy_loss=energy_loss.item(),
            loss=loss.item(), )

        # report extra information
        if self.encoder_type == "transformer" and self.use_scaled_pos_enc:
            stats.update(encoder_alpha=self.encoder.embed[-1].alpha.item(), )
        if self.decoder_type == "transformer" and self.use_scaled_pos_enc:
            stats.update(decoder_alpha=self.decoder.embed[-1].alpha.item(), )

        # loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats

    def _forward(
            self,
            xs: paddle.Tensor,
            ilens: paddle.Tensor,
            ys: paddle.Tensor=None,
            olens: paddle.Tensor=None,
            ds: paddle.Tensor=None,
            ps: paddle.Tensor=None,
            es: paddle.Tensor=None,
            is_inference: bool=False,
            alpha: float=1.0, ) -> Sequence[paddle.Tensor]:
        # forward encoder
        x_masks = self._source_mask(ilens)
        print("xs.shape in fastspeech2.py:", xs.shape)
        hs, _ = self.encoder(xs, x_masks)  # (B, Tmax, adim)

        # forward duration predictor and variance predictors
        d_masks = make_pad_mask(ilens)

        if self.stop_gradient_from_pitch_predictor:
            p_outs = self.pitch_predictor(hs.detach(), d_masks.unsqueeze(-1))
        else:
            p_outs = self.pitch_predictor(hs, d_masks.unsqueeze(-1))
        if self.stop_gradient_from_energy_predictor:
            e_outs = self.energy_predictor(hs.detach(), d_masks.unsqueeze(-1))
        else:
            e_outs = self.energy_predictor(hs, d_masks.unsqueeze(-1))
        print("p_outs.shape:", p_outs.shape)
        if is_inference:
            d_outs = self.duration_predictor.inference(hs,
                                                       d_masks)  # (B, Tmax)
            # use prediction in inference
            # (B, Tmax, 1)

            p_embs = self.pitch_embed(p_outs.transpose((0, 2, 1))).transpose(
                (0, 2, 1))
            e_embs = self.energy_embed(e_outs.transpose((0, 2, 1))).transpose(
                (0, 2, 1))
            hs = hs + e_embs + p_embs
            hs = self.length_regulator(hs, d_outs, alpha)  # (B, Lmax, adim)
        else:
            d_outs = self.duration_predictor(hs, d_masks)
            # use groundtruth in training
            print("ps.shape:", ps.shape)
            p_embs = self.pitch_embed(ps.transpose((0, 2, 1))).transpose(
                (0, 2, 1))
            e_embs = self.energy_embed(es.transpose((0, 2, 1))).transpose(
                (0, 2, 1))
            hs = hs + e_embs + p_embs
            hs = self.length_regulator(hs, ds)  # (B, Lmax, adim)

        # forward decoder
        if olens is not None and not is_inference:
            if self.reduction_factor > 1:
                # 直接to_paddle ,维度会增加 1,需要先转成 numpy
                olens_in = paddle.to_tensor(
                    [olen // self.reduction_factor for olen in olens.numpy()])
            else:
                olens_in = olens
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None
        zs, _ = self.decoder(hs, h_masks)  # (B, Lmax, adim)
        before_outs = self.feat_out(zs).reshape(
            (zs.shape[0], -1, self.odim))  # (B, Lmax, odim)

        # postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose((0, 2, 1))).transpose((0, 2, 1))

        return before_outs, after_outs, d_outs, p_outs, e_outs

    def inference(
            self,
            text: paddle.Tensor,
            speech: paddle.Tensor=None,
            durations: paddle.Tensor=None,
            pitch: paddle.Tensor=None,
            energy: paddle.Tensor=None,
            alpha: float=1.0,
            use_teacher_forcing: bool=False, ) -> Tuple[
                paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T,).
            speech (Tensor, optional): Feature sequence to extract style (N, idim).
            durations (LongTensor, optional): Groundtruth of duration (T + 1,).
            pitch (Tensor, optional): Groundtruth of token-averaged pitch (T + 1, 1).
            energy (Tensor, optional): Groundtruth of token-averaged energy (T + 1, 1).
            alpha (float, optional): Alpha to control the speed.
            use_teacher_forcing (bool, optional): Whether to use teacher forcing.
                If true, groundtruth of duration, pitch and energy will be used.

        Returns:
            Tensor: Output sequence of features (L, odim).
            None: Dummy for compatibility.
            None: Dummy for compatibility.

        """
        x, y = text, speech
        d, p, e = durations, pitch, energy

        # add eos at the last of sequence
        x = np.pad(text.numpy(),
                   pad_width=((0, 1)),
                   mode="constant",
                   constant_values=self.padding_idx)
        x = paddle.to_tensor(x)

        # setup batch axis
        ilens = paddle.to_tensor(
            [x.shape[0]], dtype=paddle.int64, place=x.place)
        xs, ys = x.unsqueeze(0), None
        if y is not None:
            ys = y.unsqueeze(0)

        if use_teacher_forcing:
            # use groundtruth of duration, pitch, and energy
            ds, ps, es = d.unsqueeze(0), p.unsqueeze(0), e.unsqueeze(0)
            _, outs, *_ = self._forward(
                xs,
                ilens,
                ys,
                ds=ds,
                ps=ps,
                es=es, )  # (1, L, odim)
        else:
            _, outs, *_ = self._forward(
                xs,
                ilens,
                ys,
                is_inference=True,
                alpha=alpha, )  # (1, L, odim)

        return outs[0], None, None

    def _source_mask(self, ilens: paddle.Tensor) -> paddle.Tensor:
        """Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                dtype=paddle.bool

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 0]]]) bool

        """
        x_masks = make_non_pad_mask(ilens)
        return x_masks.unsqueeze(-2)


class FastSpeech2Loss(nn.Layer):
    """Loss function module for FastSpeech2."""

    def __init__(self,
                 use_masking: bool=True,
                 use_weighted_masking: bool=False):
        """Initialize feed-forward Transformer loss module.

        Args:
            use_masking (bool):
                Whether to apply masking for padded part in loss calculation.
            use_weighted_masking (bool):
                Whether to weighted masking in loss calculation.

        """
        assert check_argument_types()
        super().__init__()

        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = nn.L1Loss(reduction=reduction)
        self.mse_criterion = nn.MSELoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)

    def forward(
            self,
            after_outs: paddle.Tensor,
            before_outs: paddle.Tensor,
            d_outs: paddle.Tensor,
            p_outs: paddle.Tensor,
            e_outs: paddle.Tensor,
            ys: paddle.Tensor,
            ds: paddle.Tensor,
            ps: paddle.Tensor,
            es: paddle.Tensor,
            ilens: paddle.Tensor,
            olens: paddle.Tensor, ) -> Tuple[paddle.Tensor, paddle.Tensor,
                                             paddle.Tensor, paddle.Tensor]:
        """Calculate forward propagation.

        Args:
            after_outs (Tensor): Batch of outputs after postnets (B, Lmax, odim).
            before_outs (Tensor): Batch of outputs before postnets (B, Lmax, odim).
            d_outs (LongTensor): Batch of outputs of duration predictor (B, Tmax).
            p_outs (Tensor): Batch of outputs of pitch predictor (B, Tmax, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, Tmax, 1).
            ys (Tensor): Batch of target features (B, Lmax, odim).
            ds (LongTensor): Batch of durations (B, Tmax).
            ps (Tensor): Batch of target token-averaged pitch (B, Tmax, 1).
            es (Tensor): Batch of target token-averaged energy (B, Tmax, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).
            olens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.
            Tensor: Energy predictor loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            before_outs = before_outs.masked_select(out_masks)
            if after_outs is not None:
                after_outs = after_outs.masked_select(out_masks)
            ys = ys.masked_select(out_masks)
            duration_masks = make_non_pad_mask(ilens).to(ys.device)
            d_outs = d_outs.masked_select(duration_masks)
            ds = ds.masked_select(duration_masks)
            pitch_masks = make_non_pad_mask(ilens).unsqueeze(-1).to(ys.device)
            p_outs = p_outs.masked_select(pitch_masks)
            e_outs = e_outs.masked_select(pitch_masks)
            ps = ps.masked_select(pitch_masks)
            es = es.masked_select(pitch_masks)

        # calculate loss
        l1_loss = self.l1_criterion(before_outs, ys)
        if after_outs is not None:
            l1_loss += self.l1_criterion(after_outs, ys)
        duration_loss = self.duration_criterion(d_outs, ds)
        pitch_loss = self.mse_criterion(p_outs, ps)
        energy_loss = self.mse_criterion(e_outs, es)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(olens).unsqueeze(-1)
            out_weights = out_masks.cast(
                dtype=paddle.float32) / out_masks.cast(
                    dtype=paddle.float32).sum(axis=1, keepdim=True)
            out_weights /= ys.shape[0] * ys.shape[2]
            duration_masks = make_non_pad_mask(ilens)
            duration_weights = (duration_masks.cast(dtype=paddle.float32) /
                                duration_masks.cast(dtype=paddle.float32).sum(
                                    axis=1, keepdim=True))
            duration_weights /= ds.shape[0]

            # apply weight

            l1_loss = l1_loss.multiply(out_weights)
            l1_loss = l1_loss.masked_select(
                out_masks.broadcast_to(l1_loss.shape)).sum()
            duration_loss = (duration_loss.multiply(duration_weights)
                             .masked_select(duration_masks).sum())
            pitch_masks = duration_masks.unsqueeze(-1)
            pitch_weights = duration_weights.unsqueeze(-1)
            pitch_loss = pitch_loss.multiply(pitch_weights)
            pitch_loss = pitch_loss.masked_select(
                pitch_masks.broadcast_to(pitch_loss.shape)).sum()
            energy_loss = energy_loss.multiply(pitch_weights)
            energy_loss = energy_loss.masked_select(
                pitch_masks.broadcast_to(energy_loss.shape)).sum()

        return l1_loss, duration_loss, pitch_loss, energy_loss
