import numba
import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F

def weighted_mean(input, weight):
    """weighted mean.(It can also be used as masked mean.)

    Args:
        input (Tensor): input tensor, floating point dtype.
        weight (Tensor): weight tensor with broadcastable shape.

    Returns:
        Tensor: shape(1,), weighted mean tensor with the same dtype as input.
    """
    weight = paddle.cast(weight, input.dtype) 
    return paddle.mean(input * weight)

def masked_l1_loss(prediction, target, mask):
    abs_error = F.l1_loss(prediction, target, reduction='none')
    return weighted_mean(abs_error, mask)

def masked_softmax_with_cross_entropy(logits, label, mask, axis=-1):
    ce = F.softmax_with_cross_entropy(logits, label, axis=axis)
    return weighted_mean(ce, mask)

def diagonal_loss(attentions, input_lengths, target_lengths, g=0.2, multihead=False):
    """A metric to evaluate how diagonal a attention distribution is."""
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