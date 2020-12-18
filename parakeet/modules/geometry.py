import numpy as np
import paddle

def shuffle_dim(x, axis, perm=None):
    """Permute input tensor along aixs given the permutation or randomly.

    Parameters
    ----------
        x : Tensor
            The input tensor.
            
        axis : int
            The axis to shuffle.
            
        perm : List[int], ndarray, optional
            The order to reorder the tensor along the `axis`-th dimension.
            
            It is a permutation of ``[0, d)``, where d is the size of the 
            ``axis``-th dimension of the input tensor. If not provided, 
            a random permutation is used. Defaults to None.

    Returns
    ---------
    Tensor
        The shuffled tensor, which has the same shape as x does.
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
