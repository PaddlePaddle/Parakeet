import paddle
from paddle import fluid
import paddle.fluid.dygraph as dg
import numpy as np

import weight_norm


def Embedding(name_scope,
              num_embeddings,
              embed_dim,
              padding_idx=None,
              std=0.1,
              dtype="float32"):
    # param attrs
    weight_attr = fluid.ParamAttr(initializer=fluid.initializer.Normal(
        scale=std))
    layer = dg.Embedding(
        name_scope, (num_embeddings, embed_dim),
        padding_idx=padding_idx,
        param_attr=weight_attr,
        dtype=dtype)
    return layer


def FC(name_scope,
       in_features,
       size,
       num_flatten_dims=1,
       relu=False,
       dropout=0.0,
       act=None,
       dtype="float32"):
    """
    A special Linear Layer, when it is used with dropout, the weight is 
    initialized as normal(0, std=np.sqrt((1-dropout) / in_features))
    """

    # stds
    if isinstance(in_features, int):
        in_features = [in_features]

    stds = [np.sqrt((1.0 - dropout) / in_feature) for in_feature in in_features]
    if relu:
        stds = [std * np.sqrt(2.0) for std in stds]

    weight_inits = [
        fluid.initializer.NormalInitializer(scale=std) for std in stds
    ]
    bias_init = fluid.initializer.ConstantInitializer(0.0)

    # param attrs
    weight_attrs = [fluid.ParamAttr(initializer=init) for init in weight_inits]
    bias_attr = fluid.ParamAttr(initializer=bias_init)

    layer = weight_norm.FC(name_scope,
                           size,
                           num_flatten_dims=num_flatten_dims,
                           param_attr=weight_attrs,
                           bias_attr=bias_attr,
                           act=act,
                           dtype=dtype)
    return layer


def Conv1D(name_scope,
           in_channels,
           num_filters,
           filter_size=2,
           dilation=1,
           groups=None,
           causal=False,
           std_mul=1.0,
           dropout=0.0,
           use_cudnn=True,
           act=None,
           dtype="float32"):
    """
    A special Conv1D Layer, when it is used with dropout, the weight is 
    initialized as 
    normal(0, std=np.sqrt(std_mul * (1-dropout) / (filter_size * in_channels)))
    """
    # std
    std = np.sqrt((std_mul * (1.0 - dropout)) / (filter_size * in_channels))
    weight_init = fluid.initializer.NormalInitializer(loc=0.0, scale=std)
    bias_init = fluid.initializer.ConstantInitializer(0.0)

    # param attrs
    weight_attr = fluid.ParamAttr(initializer=weight_init)
    bias_attr = fluid.ParamAttr(initializer=bias_init)

    layer = weight_norm.Conv1D(
        name_scope,
        num_filters,
        filter_size,
        dilation,
        groups=groups,
        causal=causal,
        param_attr=weight_attr,
        bias_attr=bias_attr,
        use_cudnn=use_cudnn,
        act=act,
        dtype=dtype)
    return layer


class Conv1D_GU(dg.Layer):
    def __init__(self,
                 name_scope,
                 conditioner_dim,
                 in_channels,
                 num_filters,
                 filter_size,
                 dilation,
                 causal=False,
                 residual=True,
                 dtype="float32"):
        super(Conv1D_GU, self).__init__(name_scope, dtype=dtype)

        self.conditioner_dim = conditioner_dim
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.dilation = dilation
        self.causal = causal
        self.residual = residual

        if residual:
            assert (
                in_channels == num_filters
            ), "this block uses residual connection"\
                "the input_channels should equals num_filters"

        self.conv = Conv1D(
            self.full_name(),
            in_channels,
            2 * num_filters,
            filter_size,
            dilation,
            causal=causal,
            dtype=dtype)

        self.fc = Conv1D(
            self.full_name(),
            conditioner_dim,
            2 * num_filters,
            filter_size=1,
            dilation=1,
            causal=False,
            dtype=dtype)

    def forward(self, x, skip=None, conditioner=None):
        """
        Args:
            x (Variable): Shape(B, C_in, 1, T), the input of Conv1DGLU
                layer, where B means batch_size, C_in means the input channels
                T means input time steps.
            conditioner (Variable): Shape(B, C_con, 1, T), expanded mel
                conditioner, where C_con is conditioner hidden dim which
                equals the num of mel bands. Note that when using residual
                connection, the Conv1DGLU does not change the number of
                channels, so out channels equals input channels.
        Returns:
            x (Variable): Shape(B, C_out, 1, T), the output of Conv1DGLU, where
                C_out means the output channels of Conv1DGLU.
        """
        residual = x
        x = self.conv(x)

        if conditioner is not None:
            cond_bias = self.fc(conditioner)
            x += cond_bias

        content, gate = fluid.layers.split(x, num_or_sections=2, dim=1)

        # Gated Unit.
        x = fluid.layers.elementwise_mul(fluid.layers.sigmoid(gate),
                                         fluid.layers.tanh(content))

        if skip is None:
            skip = x
        else:
            skip = fluid.layers.scale(skip + x, np.sqrt(0.5))

        if self.residual:
            x = fluid.layers.scale(residual + x, np.sqrt(0.5))

        return x, skip

    def add_input(self, x, skip=None, conditioner=None):
        """
        Inputs:
        x: shape(B, num_filters, 1, time_steps)
        conditioner: shape(B, conditioner_dim, 1, time_steps)
        Outputs:
        out: shape(B, num_filters, 1, time_steps), where time_steps = 1
        """
        residual = x

        # add step input and produce step output
        x = self.conv.add_input(x)

        if conditioner is not None:
            cond_bias = self.fc(conditioner)
            x += cond_bias

        content, gate = fluid.layers.split(x, num_or_sections=2, dim=1)

        # Gated Unit.
        x = fluid.layers.elementwise_mul(fluid.layers.sigmoid(gate),
                                         fluid.layers.tanh(content))

        if skip is None:
            skip = x
        else:
            skip = fluid.layers.scale(skip + x, np.sqrt(0.5))

        if self.residual:
            x = fluid.layers.scale(residual + x, np.sqrt(0.5))

        return x, skip


def Conv2DTranspose(name_scope,
                    num_filters,
                    filter_size,
                    padding=0,
                    stride=1,
                    dilation=1,
                    use_cudnn=True,
                    act=None,
                    dtype="float32"):
    val = 1.0 / (filter_size[0] * filter_size[1])
    weight_init = fluid.initializer.ConstantInitializer(val)
    weight_attr = fluid.ParamAttr(initializer=weight_init)

    layer = weight_norm.Conv2DTranspose(
        name_scope,
        num_filters,
        filter_size=filter_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        param_attr=weight_attr,
        use_cudnn=use_cudnn,
        act=act,
        dtype=dtype)

    return layer
