import math
import numpy as np
import paddle
from paddle.nn import functional as F


def positional_encoding(start_index, length, size, dtype=None):
    """
    Generate standard positional encoding.
    
    pe(pos, 2i) = sin(pos / 10000 ** (2i / size))
    pe(pos, 2i+1) = cos(pos / 10000 ** (2i / size))
    
    Args:
        start_index (int): the start index.
        length (int): the length of the positional encoding.
        size (int): positional encoding dimension.
    
    Returns:
        encodings (Tensor): shape(length, size), the positional encoding.
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
