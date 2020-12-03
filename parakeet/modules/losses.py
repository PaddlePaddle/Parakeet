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
    broadcast_factor = input.numel() / weight.numel()
    return paddle.sum(input * weight) / (paddle.sum(weight) * broadcast_factor)

def masked_l1_loss(prediction, target, mask):
    abs_error = F.l1_loss(prediction, target, reduction='none')
    return weighted_mean(abs_error, mask)

def masked_softmax_with_cross_entropy(logits, label, mask, axis=-1):
    ce = F.softmax_with_cross_entropy(logits, label, axis=axis)
    return weighted_mean(ce, mask)
