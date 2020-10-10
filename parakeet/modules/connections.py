import paddle
from paddle import nn
from paddle.nn import functional as F

def residual_connection(input, layer):
    """residual connection, only used for single input-single output layer.
    y = x + F(x) where F corresponds to the layer.

    Args:
        x (Tensor): the input tensor.
        layer (callable): a callable that preserve tensor shape.
    """
    return input + layer(input)

class ResidualWrapper(nn.Layer):
    def __init__(self, layer):
        super(ResidualWrapper, self).__init__()
        self.layer = layer
    
    def forward(self, x):
        return residual_connection(x, self.layer)


class PreLayerNormWrapper(nn.Layer):
    def __init__(self, layer, d_model):
        super(PreLayerNormWrapper, self).__init__()
        self.layer = layer
        self.layer_norm = nn.LayerNorm([d_model], epsilon=1e-6)
    
    def forward(self, x):
        return x + self.layer(self.layer_norm(x))


class PostLayerNormWrapper(nn.Layer):
    def __init__(self, layer, d_model):
        super(PostLayerNormWrapper, self).__init__()
        self.layer = layer
        self.layer_norm = nn.LayerNorm([d_model], epsilon=1e-6)
    
    def forward(self, x):
        return self.layer_norm(x + self.layer(x))


def context_gate(input, axis):
    """sigmoid gate the content by gate.

    Args:
        input (Tensor): shape(*, d_axis, *), the input, treated as content & gate.
        axis (int): the axis to chunk content and gate.

    Raises:
        ValueError: if input.shape[axis] is not even.

    Returns:
        Tensor: shape(*, d_axis / 2 , *), the gated content.
    """
    size = input.shape[axis]
    if size % 2 != 0:
        raise ValueError("the size of the {}-th dimension of input should "
                         "be even, but received {}".format(axis, size))
    content, gate = paddle.chunk(input, 2, axis)
    return F.sigmoid(gate) * content
