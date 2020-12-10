import paddle
from paddle import distributed as dist
from functools import wraps

__all__ = ["rank_zero_only"]


def rank_zero_only(func):
    local_rank = dist.get_rank()

    @wraps(func)
    def wrapper(*args, **kwargs):
        if local_rank != 0:
            return 
        result = func(*args, **kwargs)
        return result
    
    return wrapper



