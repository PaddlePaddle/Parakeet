import paddle
import paddle.fluid.dygraph as dg

import numpy as np
from paddle import fluid
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as F
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype


def l2_norm(x, axis, epsilon=1e-12, name=None):
    if len(x.shape) == 1:
        axis = 0
    check_variable_and_dtype(x, "X", ("float32", "float64"), "norm")

    helper = LayerHelper("l2_normalize", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    norm = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="norm",
        inputs={"X": x},
        outputs={"Out": out,
                 "Norm": norm},
        attrs={
            "axis": 1 if axis is None else axis,
            "epsilon": epsilon,
        })
    return F.squeeze(norm, axes=[axis])
    
def norm_except_dim(p, dim):
    shape = p.shape
    ndims = len(shape)
    if dim is None:
        return F.sqrt(F.reduce_sum(F.square(p)))
    elif dim == 0:
        p_matrix = F.reshape(p, (shape[0], -1))
        return l2_norm(p_matrix, axis=1)
    elif dim == -1 or dim == ndims - 1:
        p_matrix = F.reshape(p, (-1, shape[-1]))
        return l2_norm(p_matrix, axis=0)
    else:
        perm = list(range(ndims))
        perm[0] = dim
        perm[dim] = 0
        p_transposed = F.transpose(p, perm)
        return norm_except_dim(p_transposed, 0)

def _weight_norm(v, g, dim):
    shape = v.shape
    ndims = len(shape)

    if dim is None:
        v_normalized = v / (F.sqrt(F.reduce_sum(F.square(v))) + 1e-12)
    elif dim == 0:
        p_matrix = F.reshape(v, (shape[0], -1))
        v_normalized = F.l2_normalize(p_matrix, axis=1)
        v_normalized = F.reshape(v_normalized, shape)
    elif dim == -1 or dim == ndims - 1:
        p_matrix = F.reshape(v, (-1, shape[-1]))
        v_normalized = F.l2_normalize(p_matrix, axis=0)
        v_normalized = F.reshape(v_normalized, shape)
    else:
        perm = list(range(ndims))
        perm[0] = dim
        perm[dim] = 0
        p_transposed = F.transpose(v, perm)
        transposed_shape = p_transposed.shape
        p_matrix = F.reshape(p_transposed, (p_transposed.shape[0], -1))
        v_normalized = F.l2_normalize(p_matrix, axis=1)
        v_normalized = F.reshape(v_normalized, transposed_shape)
        v_normalized = F.transpose(v_normalized, perm)
    weight = F.elementwise_mul(v_normalized, g, axis=dim if dim is not None else -1)
    return weight


class WeightNorm(object):
    def __init__(self, name, dim):
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    def compute_weight(self, module):
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        w = _weight_norm(v, g, self.dim)
        return w

    @staticmethod
    def apply(module: dg.Layer, name, dim):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError("Cannot register two weight_norm hooks on "
                                   "the same parameter {}".format(name))

        if dim is None:
            dim = -1

        fn = WeightNorm(name, dim)

        # remove w from parameter list
        w = getattr(module, name)
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        g_var = norm_except_dim(w, dim)
        v = module.create_parameter(w.shape, dtype=w.dtype)
        module.add_parameter(name + "_v", v)
        g = module.create_parameter(g_var.shape, dtype=g_var.dtype)
        module.add_parameter(name + "_g", g)
        with dg.no_grad():
            F.assign(w, v)
            F.assign(g_var, g)
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)
        return fn

    def remove(self, module):
        w_var = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_g']
        del module._parameters[self.name + '_v']
        w = module.create_parameter(w_var.shape, dtype=w_var.dtype)
        module.add_parameter(self.name, w)
        with dg.no_grad():
            F.assign(w_var, w)

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def weight_norm(module, name='weight', dim=0):
    WeightNorm.apply(module, name, dim)
    return module


def remove_weight_norm(module, name='weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))