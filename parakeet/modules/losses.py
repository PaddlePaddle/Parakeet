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

import numba
import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F

__all__ = [
    "weighted_mean",
    "masked_l1_loss",
    "masked_softmax_with_cross_entropy",
    "diagonal_loss",
]


def weighted_mean(input, weight):
    """Weighted mean. It can also be used as masked mean.

    Parameters
    -----------
    input : Tensor 
        The input tensor.
    weight : Tensor
        The weight tensor with broadcastable shape with the input.

    Returns
    ----------
    Tensor [shape=(1,)]
        Weighted mean tensor with the same dtype as input.
        
    Warnings
    ---------
    This is not a mathematical weighted mean. It performs weighted sum and 
    simple average.
    """
    weight = paddle.cast(weight, input.dtype)
    return paddle.mean(input * weight)


def masked_l1_loss(prediction, target, mask):
    """Compute maksed L1 loss.

    Parameters
    ----------
    prediction : Tensor
        The prediction.
        
    target : Tensor
        The target. The shape should be broadcastable to ``prediction``.
        
    mask : Tensor
        The mask. The shape should be broadcatable to the broadcasted shape of 
        ``prediction`` and ``target``.

    Returns
    -------
    Tensor [shape=(1,)]
        The masked L1 loss.
    """
    abs_error = F.l1_loss(prediction, target, reduction='none')
    loss = weighted_mean(abs_error, mask)
    return loss


def masked_softmax_with_cross_entropy(logits, label, mask, axis=-1):
    """Compute masked softmax with cross entropy loss.

    Parameters
    ----------
    logits : Tensor
        The logits. The ``axis``-th axis is the class dimension.
        
    label : Tensor [dtype: int]
        The label. The size of the ``axis``-th axis should be 1.
        
    mask : Tensor 
        The mask. The shape should be broadcastable to ``label``.
        
    axis : int, optional
        The index of the class dimension in the shape of ``logits``, by default
        -1.

    Returns
    -------
    Tensor [shape=(1,)]
        The masked softmax with cross entropy loss.
    """
    ce = F.softmax_with_cross_entropy(logits, label, axis=axis)
    loss = weighted_mean(ce, mask)
    return loss


def diagonal_loss(attentions,
                  input_lengths,
                  target_lengths,
                  g=0.2,
                  multihead=False):
    """A metric to evaluate how diagonal a attention distribution is.
    
    It is computed for batch attention distributions. For each attention 
    distribution, the valid decoder time steps and encoder time steps may
    differ.

    Parameters
    ----------
    attentions : Tensor [shape=(B, T_dec, T_enc) or (B, H, T_dec, T_dec)]
        The attention weights from an encoder-decoder structure.
        
    input_lengths : Tensor [shape=(B,)]
        The valid length for each encoder output.
        
    target_lengths : Tensor [shape=(B,)]
        The valid length for each decoder output.
        
    g : float, optional
        [description], by default 0.2.
        
    multihead : bool, optional
        A flag indicating whether ``attentions`` is a multihead attention's
        attention distribution. 
        
        If ``True``, the shape of attention is ``(B, H, T_dec, T_dec)``, by 
        default False.

    Returns
    -------
    Tensor [shape=(1,)]
        The diagonal loss.
    """
    W = guided_attentions(input_lengths, target_lengths, g)
    W_tensor = paddle.to_tensor(W)
    if not multihead:
        return paddle.mean(attentions * W_tensor)
    else:
        return paddle.mean(attentions * paddle.unsqueeze(W_tensor, 1))


@numba.jit(nopython=True)
def guided_attention(N, max_N, T, max_T, g):
    W = np.zeros((max_T, max_N), dtype=np.float32)
    for t in range(T):
        for n in range(N):
            W[t, n] = 1 - np.exp(-(n / N - t / T)**2 / (2 * g * g))
    # (T_dec, T_enc)
    return W


def guided_attentions(input_lengths, target_lengths, g=0.2):
    B = len(input_lengths)
    max_input_len = input_lengths.max()
    max_target_len = target_lengths.max()
    W = np.zeros((B, max_target_len, max_input_len), dtype=np.float32)
    for b in range(B):
        W[b] = guided_attention(input_lengths[b], max_input_len,
                                target_lengths[b], max_target_len, g)
    # (B, T_dec, T_enc)
    return W
