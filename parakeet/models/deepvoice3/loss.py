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
import numpy as np
from numba import jit

from paddle import fluid
import paddle.fluid.layers as F
import paddle.fluid.dygraph as dg


def masked_mean(inputs, mask):
    """
    Args:
        inputs (Variable): shape(B, T, C), dtype float32, the input.
        mask (Variable): shape(B, T), dtype float32, a mask. 
    Returns:
        loss (Variable): shape(1, ), dtype float32, masked mean.
    """
    channels = inputs.shape[-1]
    masked_inputs = F.elementwise_mul(inputs, mask, axis=0)
    loss = F.reduce_sum(masked_inputs) / (channels * F.reduce_sum(mask))
    return loss


@jit(nopython=True)
def guided_attention(N, max_N, T, max_T, g):
    """Generate an diagonal attention guide.
    
    Args:
        N (int): valid length of encoder.
        max_N (int): max length of encoder.
        T (int): valid length of decoder.
        max_T (int): max length of decoder.
        g (float): sigma to adjust the degree of diagonal guide.

    Returns:
        np.ndarray: shape(max_N, max_T), dtype float32, the diagonal guide.
    """
    W = np.zeros((max_N, max_T), dtype=np.float32)
    for n in range(N):
        for t in range(T):
            W[n, t] = 1 - np.exp(-(n / N - t / T)**2 / (2 * g * g))
    return W


def guided_attentions(encoder_lengths, decoder_lengths, max_decoder_len,
                      g=0.2):
    """Generate a diagonal attention guide for a batch.

    Args:
        encoder_lengths (np.ndarray): shape(B, ), dtype: int64, encoder valid lengths.
        decoder_lengths (np.ndarray): shape(B, ), dtype: int64, decoder valid lengths.
        max_decoder_len (int): max length of decoder.
        g (float, optional): sigma to adjust the degree of diagonal guide.. Defaults to 0.2.

    Returns:
        np.ndarray: shape(B, max_T, max_N), dtype float32, the diagonal guide. (max_N: max encoder length, max_T: max decoder length.)
    """
    B = len(encoder_lengths)
    max_input_len = encoder_lengths.max()
    W = np.zeros((B, max_decoder_len, max_input_len), dtype=np.float32)
    for b in range(B):
        W[b] = guided_attention(encoder_lengths[b], max_input_len,
                                decoder_lengths[b], max_decoder_len, g).T
    return W


class TTSLoss(object):
    def __init__(self,
                 masked_weight=0.0,
                 priority_bin=None,
                 priority_weight=0.0,
                 binary_divergence_weight=0.0,
                 guided_attention_sigma=0.2,
                 downsample_factor=4,
                 r=1):
        """Compute loss for Deep Voice 3 model.

        Args:
            masked_weight (float, optional): the weight of masked loss. Defaults to 0.0.
            priority_bin ([type], optional): frequency bands for linear spectrogram loss to be prioritized. Defaults to None.
            priority_weight (float, optional): weight for the prioritized frequency bands. Defaults to 0.0.
            binary_divergence_weight (float, optional): weight for binary cross entropy (used for spectrogram loss). Defaults to 0.0.
            guided_attention_sigma (float, optional): `sigma` for attention guide. Defaults to 0.2.
            downsample_factor (int, optional): the downsample factor for mel spectrogram. Defaults to 4.
            r (int, optional): frames per decoder step. Defaults to 1.
        """
        self.masked_weight = masked_weight
        self.priority_bin = priority_bin  # only used for lin-spec loss
        self.priority_weight = priority_weight  # only used for lin-spec loss
        self.binary_divergence_weight = binary_divergence_weight
        self.guided_attention_sigma = guided_attention_sigma

        self.time_shift = r
        self.r = r
        self.downsample_factor = downsample_factor

    def l1_loss(self, prediction, target, mask, priority_bin=None):
        """L1 loss for spectrogram.

        Args:
            prediction (Variable): shape(B, T, C), dtype float32, predicted spectrogram.
            target (Variable): shape(B, T, C), dtype float32, target spectrogram.
            mask (Variable): shape(B, T), mask.
            priority_bin (int, optional): frequency bands for linear spectrogram loss to be prioritized. Defaults to None.

        Returns:
            Variable: shape(1,), dtype float32, l1 loss(with mask and possibly priority bin applied.)
        """
        abs_diff = F.abs(prediction - target)

        # basic mask-weighted l1 loss
        w = self.masked_weight
        if w > 0 and mask is not None:
            base_l1_loss = w * masked_mean(abs_diff, mask) \
                         + (1 - w) * F.reduce_mean(abs_diff)
        else:
            base_l1_loss = F.reduce_mean(abs_diff)

        if self.priority_weight > 0 and priority_bin is not None:
            # mask-weighted priority channels' l1-loss
            priority_abs_diff = abs_diff[:, :, :priority_bin]
            if w > 0 and mask is not None:
                priority_loss = w * masked_mean(priority_abs_diff, mask) \
                              + (1 - w) * F.reduce_mean(priority_abs_diff)
            else:
                priority_loss = F.reduce_mean(priority_abs_diff)

            # priority weighted sum
            p = self.priority_weight
            loss = p * priority_loss + (1 - p) * base_l1_loss
        else:
            loss = base_l1_loss
        return loss

    def binary_divergence(self, prediction, target, mask):
        """Binary cross entropy loss for spectrogram. All the values in the spectrogram are treated as logits in a logistic regression.

        Args:
            prediction (Variable): shape(B, T, C), dtype float32, predicted spectrogram.
            target (Variable): shape(B, T, C), dtype float32, target spectrogram.
            mask (Variable): shape(B, T), mask.

        Returns:
            Variable: shape(1,), dtype float32, binary cross entropy loss.
        """
        flattened_prediction = F.reshape(prediction, [-1, 1])
        flattened_target = F.reshape(target, [-1, 1])
        flattened_loss = F.log_loss(
            flattened_prediction, flattened_target, epsilon=1e-8)
        bin_div = fluid.layers.reshape(flattened_loss, prediction.shape)

        w = self.masked_weight
        if w > 0 and mask is not None:
            loss = w * masked_mean(bin_div, mask) \
                 + (1 - w) * F.reduce_mean(bin_div)
        else:
            loss = F.reduce_mean(bin_div)
        return loss

    @staticmethod
    def done_loss(done_hat, done):
        """Compute done loss

        Args:
            done_hat (Variable): shape(B, T), dtype float32, predicted done probability(the probability that the final frame has been generated.)
            done (Variable): shape(B, T), dtype float32, ground truth done probability(the probability that the final frame has been generated.)

        Returns:
            Variable: shape(1, ), dtype float32, done loss.
        """
        flat_done_hat = F.reshape(done_hat, [-1, 1])
        flat_done = F.reshape(done, [-1, 1])
        loss = F.log_loss(flat_done_hat, flat_done, epsilon=1e-8)
        loss = F.reduce_mean(loss)
        return loss

    def attention_loss(self, predicted_attention, input_lengths,
                       target_lengths):
        """
        Given valid encoder_lengths and decoder_lengths, compute a diagonal guide, and compute loss from the predicted attention and the guide.
        
        Args:
            predicted_attention (Variable): shape(*, B, T_dec, T_enc), dtype float32, the alignment tensor, where B means batch size, T_dec means number of time steps of the decoder, T_enc means the number of time steps of the encoder, * means other possible dimensions.
            input_lengths (numpy.ndarray): shape(B,), dtype:int64, valid lengths (time steps) of encoder outputs.
            target_lengths (numpy.ndarray): shape(batch_size,), dtype:int64, valid lengths (time steps) of decoder outputs.
        
        Returns:
            loss (Variable): shape(1, ), dtype float32, attention loss.
        """
        n_attention, batch_size, max_target_len, max_input_len = (
            predicted_attention.shape)
        soft_mask = guided_attentions(input_lengths, target_lengths,
                                      max_target_len,
                                      self.guided_attention_sigma)
        soft_mask_ = dg.to_variable(soft_mask)
        loss = fluid.layers.reduce_mean(predicted_attention * soft_mask_)
        return loss

    def __call__(self, outputs, inputs):
        """Total loss

        Args:
            outpus is a tuple of (mel_hyp, lin_hyp, attn_hyp, done_hyp).
            mel_hyp (Variable): shape(B, T, C_mel), dtype float32, predicted mel spectrogram.
            lin_hyp (Variable): shape(B, T, C_lin), dtype float32, predicted linear spectrogram.
            done_hyp (Variable): shape(B, T), dtype float32, predicted done probability.
            attn_hyp (Variable): shape(N, B, T_dec, T_enc), dtype float32, predicted attention.

            inputs is a tuple of (mel_ref, lin_ref, done_ref, input_lengths, n_frames)
            mel_ref (Variable): shape(B, T, C_mel), dtype float32, ground truth mel spectrogram.
            lin_ref (Variable): shape(B, T, C_lin), dtype float32, ground truth linear spectrogram.
            done_ref (Variable): shape(B, T), dtype float32, ground truth done flag.
            input_lengths (Variable): shape(B, ), dtype: int, encoder valid lengths.
            n_frames (Variable): shape(B, ), dtype: int, decoder valid lengths.

        Returns:
            Dict(str, Variable): details of loss.
        """
        total_loss = 0.

        mel_hyp, lin_hyp, attn_hyp, done_hyp = outputs
        mel_ref, lin_ref, done_ref, input_lengths, n_frames = inputs

        # n_frames # mel_lengths # decoder_lengths
        max_frames = lin_hyp.shape[1]
        max_mel_steps = max_frames // self.downsample_factor
        # max_decoder_steps = max_mel_steps // self.r
        # decoder_mask = F.sequence_mask(n_frames // self.downsample_factor //
        #                                self.r,
        #                                max_decoder_steps,
        #                                dtype="float32")
        mel_mask = F.sequence_mask(
            n_frames // self.downsample_factor, max_mel_steps, dtype="float32")
        lin_mask = F.sequence_mask(n_frames, max_frames, dtype="float32")

        lin_hyp = lin_hyp[:, :-self.time_shift, :]
        lin_ref = lin_ref[:, self.time_shift:, :]
        lin_mask = lin_mask[:, self.time_shift:]
        lin_l1_loss = self.l1_loss(
            lin_hyp, lin_ref, lin_mask, priority_bin=self.priority_bin)
        lin_bce_loss = self.binary_divergence(lin_hyp, lin_ref, lin_mask)
        lin_loss = self.binary_divergence_weight * lin_bce_loss \
                    + (1 - self.binary_divergence_weight) * lin_l1_loss
        total_loss += lin_loss

        mel_hyp = mel_hyp[:, :-self.time_shift, :]
        mel_ref = mel_ref[:, self.time_shift:, :]
        mel_mask = mel_mask[:, self.time_shift:]
        mel_l1_loss = self.l1_loss(mel_hyp, mel_ref, mel_mask)
        mel_bce_loss = self.binary_divergence(mel_hyp, mel_ref, mel_mask)
        # print("=====>", mel_l1_loss.numpy()[0], mel_bce_loss.numpy()[0])
        mel_loss = self.binary_divergence_weight * mel_bce_loss \
                    + (1 - self.binary_divergence_weight) * mel_l1_loss
        total_loss += mel_loss

        attn_loss = self.attention_loss(attn_hyp,
                                        input_lengths.numpy(),
                                        n_frames.numpy() //
                                        (self.downsample_factor * self.r))
        total_loss += attn_loss

        done_loss = self.done_loss(done_hyp, done_ref)
        total_loss += done_loss

        losses = {
            "loss": total_loss,
            "mel/mel_loss": mel_loss,
            "mel/l1_loss": mel_l1_loss,
            "mel/bce_loss": mel_bce_loss,
            "lin/lin_loss": lin_loss,
            "lin/l1_loss": lin_l1_loss,
            "lin/bce_loss": lin_bce_loss,
            "done": done_loss,
            "attn": attn_loss,
        }

        return losses
