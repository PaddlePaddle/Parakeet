import numpy as np
import paddle

def shuffle_dim(x, axis, perm=None):
    """Permute input tensor along aixs given the permutation or randomly.

    Args:
        x (Tensor): shape(*, d_{axis}, *), the input tensor.
        axis (int): the axis to shuffle.
        perm (list[int], ndarray, optional): a permutation of [0, d_{axis}), 
            the order to reorder the tensor along the `axis`-th dimension, if 
            not provided, randomly shuffle the `axis`-th dimension. Defaults to 
            None.

    Returns:
        Tensor: the shuffled tensor, it has the same shape as x does.
    """
    size = x.shape[axis]
    if perm is not None and len(perm) != size:
        raise ValueError("length of permutation should equals the input "
                         "tensor's axis-th dimension's size")
    if perm is not None:
        perm = np.array(perm)
    else:
        perm = np.random.permutation(size)
    
    perm = paddle.to_tensor(perm)
    out = paddle.gather(x, perm, axis)
    return out