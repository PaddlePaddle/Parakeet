import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid import ParamAttr
from paddle.nn.initializer import Normal


def _spectral_norm(weight, weight_u, weight_v, dim, power_iters, epsilon,
                   training):
    helper = LayerHelper('_spectral_norm', **locals())
    inputs = {'Weight': weight, 'U': weight_u, 'V': weight_v}
    out = helper.create_variable_for_type_inference(weight.dtype)
    _power_iters = power_iters if training else 0
    helper.append_op(type="spectral_norm",
                     inputs=inputs,
                     outputs={
                         "Out": out,
                     },
                     attrs={
                         "dim": dim,
                         "power_iters": _power_iters,
                         "eps": epsilon,
                     })
    return out


class SpectralNorm(object):
    def __init__(self,
                 name: str = "weight",
                 power_iters: int = 1,
                 dim: int = 0,
                 epsilon: float = 1e-12):
        self.name = name
        self.dim = dim
        self.power_iters = power_iters
        self.epsilon = epsilon

    @staticmethod
    def apply(layer, name, power_iters, dim, epsilon):
        for k, hook in layer._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name))

        fn = SpectralNorm(name, power_iters, dim, epsilon)

        weight = getattr(layer, name)
        del layer._parameters[name]

        h = weight.shape[dim]
        w = np.prod(weight.shape) // h

        weight_u = layer.create_parameter(attr=ParamAttr(),
                                          shape=[h],
                                          dtype=weight.dtype,
                                          default_initializer=Normal(0., 1.))
        weight_u.trainable = False
        layer.add_parameter(name + "_u", weight_u)

        weight_v = layer.create_parameter(attr=ParamAttr(),
                                          shape=[w],
                                          dtype=weight.dtype,
                                          default_initializer=Normal(0., 1.))
        weight_v.trainable = False
        layer.add_parameter(name + "_v", weight_v)

        weight_orig = layer.create_parameter(attr=ParamAttr(),
                                             shape=weight.shape,
                                             dtype=weight.dtype)
        with paddle.no_grad():
            F.assign(weight, weight_orig)
        layer.add_parameter(name + "_orig", weight_orig)

        # call the hook once
        setattr(layer, name, fn.compute_weight(layer))

        layer.register_forward_pre_hook(fn)
        return fn

    def remove(self, layer):
        w_var = self.compute_weight(layer)

        delattr(layer, self.name)
        del layer._parameters[self.name + '_u']
        del layer._parameters[self.name + '_v']
        del layer._parameters[self.name + '_orig']
        w = layer.create_parameter(w_var.shape, dtype=w_var.dtype)
        layer.add_parameter(self.name, w)
        with paddle.no_grad():
            F.assign(w_var, w)

    def compute_weight(self, layer):
        u = getattr(layer, self.name + '_u')
        v = getattr(layer, self.name + '_v')
        orig = getattr(layer, self.name + '_orig')
        return _spectral_norm(orig,
                              u,
                              v,
                              self.dim,
                              self.power_iters,
                              self.epsilon,
                              training=layer.training)

    def __call__(self, layer, inputs):
        setattr(layer, self.name, self.compute_weight(layer))


def spectral_norm(layer,
                  name: str = "weight",
                  power_iters: int = 1,
                  dim: int = 0,
                  epsilon: float = 1e-12):
    SpectralNorm.apply(layer, name, power_iters, dim, epsilon)
    return layer


def remove_spectral_norm(layer, name='weight'):
    for k, hook in layer._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(layer)
            del layer._forward_pre_hooks[k]
            return layer

    raise ValueError("spectral_norm of '{}' not found in {}".format(
        name, layer))
