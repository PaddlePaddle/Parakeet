import numpy as np
from torch import nn
import paddle.fluid.dygraph as dg


def summary(layer):
    num_params = num_elements = 0
    print("layer summary:")
    for name, param in layer.state_dict().items():
        print("{}|{}|{}".format(name, param.shape, np.prod(param.shape)))
        num_elements += np.prod(param.shape)
        num_params += 1
    print("layer has {} parameters, {} elements.".format(
        num_params, num_elements))


def freeze(layer):
    for param in layer.parameters():
        param.trainable = False


def unfreeze(layer):
    for param in layer.parameters():
        param.trainable = True


def torch_summary(layer):
    num_params = num_elements = 0
    print("layer summary:")
    for name, param in layer.named_parameters():
        print("{}|{}|{}".format(name, param.shape, np.prod(param.shape)))
        num_elements += np.prod(param.shape)
        num_params += 1
    print("layer has {} parameters, {} elements.".format(
        num_params, num_elements))
