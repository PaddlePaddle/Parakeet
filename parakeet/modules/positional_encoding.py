import math
import paddle
from paddle.nn import functional as F

def positional_encoding(start_index, length, size, dtype=None):
    """
    Generate standard positional encoding.
    
    pe(pos, 2i) = sin(pos / 10000 ** (2i / size))
    pe(pos, 2i+1) = cos(pos / 10000 ** (2i / size))
    
    This implementation deviates from the standard implementation in that the
    sin/cos channels are not interleaved.

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
    channel = paddle.arange(0, size, 2, dtype=dtype)
    index = paddle.arange(start_index, start_index + length, 1, dtype=dtype)
    p = paddle.unsqueeze(index, -1) / (10000 ** (channel / float(size)))
    encodings = paddle.concat([paddle.sin(p), paddle.cos(p)], axis=-1)
    return encodings

def scalable_positional_encoding(start_index, length, size, omega):
    """
    A scalable positional encoding, which extends the standard positional 
    encoding by adding positioning rate (denoted as omega).
    
    pe(pos, 2i) = sin(omega * pos / 10000 ** (2i / size))
    pe(pos, 2i+1) = cos(omega * pos / 10000 ** (2i / size))
    
    This implementation deviates from the standard implementation in that the
    sin/cos channels are not interleaved.
    
    Args:
        start_index (int): the start index.
        length (int): the length of the positional encoding.
        size (int): positional encoding dimension.
        omgea (Tensor): shape(batch_size, ), positional rates.

    Returns:
        encodings: shape(batch_size, length, size), position embedding, the 
        data type is the same as omega.
    """
    dtype = omega.dtype
    index = paddle.arange(start_index, start_index + length, 1, dtype=dtype)
    channel = paddle.arange(0, size, 2, dtype=dtype)

    p = paddle.unsqueeze(omega, [1, 2]) \
      * paddle.unsqueeze(index, [1]) \
      / (10000 ** (channel / float(size)))

    encodings = paddle.concat([paddle.sin(p), paddle.cos(p)], axis=-1)
    return encodings
