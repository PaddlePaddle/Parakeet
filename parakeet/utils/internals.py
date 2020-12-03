import numpy as np
from paddle.framework import core

def convert_dtype_to_np_dtype_(dtype):
    """
    Convert paddle's data type to corrsponding numpy data type.

    Args:
        dtype(np.dtype): the data type in paddle.

    Returns:
        type: the data type in numpy.

    """
    if dtype is core.VarDesc.VarType.FP32:
        return np.float32
    elif dtype is core.VarDesc.VarType.FP64:
        return np.float64
    elif dtype is core.VarDesc.VarType.FP16:
        return np.float16
    elif dtype is core.VarDesc.VarType.BOOL:
        return np.bool
    elif dtype is core.VarDesc.VarType.INT32:
        return np.int32
    elif dtype is core.VarDesc.VarType.INT64:
        return np.int64
    elif dtype is core.VarDesc.VarType.INT16:
        return np.int16
    elif dtype is core.VarDesc.VarType.INT8:
        return np.int8
    elif dtype is core.VarDesc.VarType.UINT8:
        return np.uint8
    elif dtype is core.VarDesc.VarType.BF16:
        return np.uint16
    else:
        raise ValueError("Not supported dtype %s" % dtype)
