import math
import numpy as np
import paddle
from paddle.nn import functional as F

__all__ = ["positional_encoding"]

def positional_encoding(start_index, length, size, dtype=None):
    r"""Generate standard positional encoding matrix.
    
    .. math::
    
        pe(pos, 2i) = sin(\frac{pos}{10000^{\frac{2i}{size}}}) \\
        pe(pos, 2i+1) = cos(\frac{pos}{10000^{\frac{2i}{size}}})
    
    Parameters
    ----------
    start_index : int
        The start index.
    length : int
        The timesteps of the positional encoding to generate.
    size : int 
        Feature size of positional encoding.
    
    Returns
    -------
    Tensor [shape=(length, size)]
        The positional encoding.
        
    Raises
    ------
    ValueError
        If ``size`` is not divisible by 2.
    """
    if (size % 2 != 0):
        raise ValueError("size should be divisible by 2")
    dtype = dtype or paddle.get_default_dtype()
    channel = np.arange(0, size, 2)
    index = np.arange(start_index, start_index + length, 1)
    p = np.expand_dims(index, -1) / (10000 ** (channel / float(size)))
    encodings = np.zeros([length, size])
    encodings[:, 0::2] = np.sin(p)
    encodings[:, 1::2] = np.cos(p)
    encodings = paddle.to_tensor(encodings)
    return encodings
