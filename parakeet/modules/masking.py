import paddle
from paddle.fluid.layers import sequence_mask

def id_mask(input, padding_index=0, dtype="bool"):
    return paddle.cast(input != padding_index, dtype)

def feature_mask(input, axis, dtype="bool"):
    feature_sum = paddle.sum(paddle.abs(input), axis=axis, keepdim=True)
    return paddle.cast(feature_sum != 0, dtype)

def combine_mask(padding_mask, no_future_mask):
    """
    Combine the padding mask and no future mask for transformer decoder. 
    Padding mask is used to mask padding positions and no future mask is used 
    to prevent the decoder to see future information.

    Args:
        padding_mask (Tensor): shape(batch_size, time_steps), dtype: float32 or float64, decoder padding mask. 
        no_future_mask (Tensor): shape(time_steps, time_steps), dtype: float32 or float64, no future mask.

    Returns:
        Tensor: shape(batch_size, time_steps, time_steps), combined mask.
    """
    # TODO: to support boolean mask by using logical_and?
    if padding_mask.dtype == paddle.fluid.core.VarDesc.VarType.BOOL:
        return paddle.logical_and(padding_mask, no_future_mask)
    else:
        return padding_mask * no_future_mask

def future_mask(time_steps, dtype="bool"):
    mask = paddle.tril(paddle.ones([time_steps, time_steps]))
    return paddle.cast(mask, dtype)
