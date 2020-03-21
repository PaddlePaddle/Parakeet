# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle import fluid
import paddle.fluid.layers as F
import paddle.fluid.dygraph as dg


class Pool1D(dg.Layer):
    """
    A Pool 1D block implemented with Pool2D.
    """

    def __init__(self,
                 pool_size=-1,
                 pool_type='max',
                 pool_stride=1,
                 pool_padding=0,
                 global_pooling=False,
                 use_cudnn=True,
                 ceil_mode=False,
                 exclusive=True,
                 data_format='NCT'):
        super(Pool1D, self).__init__()
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.global_pooling = global_pooling
        self.use_cudnn = use_cudnn
        self.ceil_mode = ceil_mode
        self.exclusive = exclusive
        self.data_format = data_format

        self.pool2d = dg.Pool2D(
            [1, pool_size],
            pool_type=pool_type,
            pool_stride=[1, pool_stride],
            pool_padding=[0, pool_padding],
            global_pooling=global_pooling,
            use_cudnn=use_cudnn,
            ceil_mode=ceil_mode,
            exclusive=exclusive)

    def forward(self, x):
        """
        Args:
            x (Variable): Shape(B, C_in, 1, T), the input, where C_in means
                input channels.
        Returns:
            x (Variable): Shape(B, C_out, 1, T), the outputs, where C_out means
                output channels (num_filters).
        """
        if self.data_format == 'NTC':
            x = fluid.layers.transpose(x, [0, 2, 1])
        x = fluid.layers.unsqueeze(x, [2])
        x = self.pool2d(x)
        x = fluid.layers.squeeze(x, [2])
        if self.data_format == 'NTC':
            x = fluid.layers.transpose(x, [0, 2, 1])
        return x


class Conv1D(dg.Conv2D):
    """A standard Conv1D layer that use (B, C, T) data layout. It inherit Conv2D and 
    use (B, C, 1, T) data layout to compute 1D convolution. Nothing more.
    NOTE: we inherit Conv2D instead of encapsulate a Conv2D layer to make it a simple
    layer, instead of a complex one. So we can easily apply weight norm to it.
    """

    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 param_attr=None,
                 bias_attr=None,
                 use_cudnn=True,
                 act=None,
                 dtype='float32'):
        super(Conv1D, self).__init__(
            num_channels,
            num_filters, (1, filter_size),
            stride=(1, stride),
            padding=(0, padding),
            dilation=(1, dilation),
            groups=groups,
            param_attr=param_attr,
            bias_attr=bias_attr,
            use_cudnn=use_cudnn,
            act=act,
            dtype=dtype)

    def forward(self, x):
        """Compute Conv1D by unsqueeze the input and squeeze the output.

        Args:
            x (Variable): shape(B, C_in, T_in), dtype float32, input of Conv1D.

        Returns:
            Variable: shape(B, C_out, T_out), dtype float32, output of Conv1D.
        """
        x = F.unsqueeze(x, [2])
        x = super(Conv1D, self).forward(x)  # maybe risky here
        x = F.squeeze(x, [2])
        return x


class Conv1DTranspose(dg.Conv2DTranspose):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 param_attr=None,
                 bias_attr=None,
                 use_cudnn=True,
                 act=None,
                 dtype='float32'):
        super(Conv1DTranspose, self).__init__(
            num_channels,
            num_filters, (1, filter_size),
            output_size=None,
            padding=(0, padding),
            stride=(1, stride),
            dilation=(1, dilation),
            groups=groups,
            param_attr=param_attr,
            bias_attr=bias_attr,
            use_cudnn=use_cudnn,
            act=act,
            dtype=dtype)

    def forward(self, x):
        """Compute Conv1DTranspose by unsqueeze the input and squeeze the output.

        Args:
            x (Variable): shape(B, C_in, T_in), dtype float32, input of Conv1DTranspose.

        Returns:
            Variable: shape(B, C_out, T_out), dtype float32, output of Conv1DTranspose.
        """
        x = F.unsqueeze(x, [2])
        x = super(Conv1DTranspose, self).forward(x)  # maybe risky here
        x = F.squeeze(x, [2])
        return x


class Conv1DCell(Conv1D):
    """A causal convolve-1d cell. It uses causal padding, padding(receptive_field -1,  0).
    But Conv2D in dygraph does not support asymmetric padding yet, we just pad
    (receptive_field -1, receptive_field -1) and drop last receptive_field -1 steps in 
    the output.
    
    It is a cell that it acts like an RNN cell. It does not support stride > 1, and it
    ensures 1-to-1 mapping from input time steps to output timesteps.
    """

    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 dilation=1,
                 causal=False,
                 groups=1,
                 param_attr=None,
                 bias_attr=None,
                 use_cudnn=True,
                 act=None,
                 dtype='float32'):
        receptive_field = 1 + dilation * (filter_size - 1)
        padding = receptive_field - 1 if causal else receptive_field // 2
        self._receptive_field = receptive_field
        self.causal = causal
        super(Conv1DCell, self).__init__(
            num_channels,
            num_filters,
            filter_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=groups,
            param_attr=param_attr,
            bias_attr=bias_attr,
            use_cudnn=use_cudnn,
            act=act,
            dtype=dtype)

    def forward(self, x):
        """Compute Conv1D by unsqueeze the input and squeeze the output.

        Args:
            x (Variable): shape(B, C_in, T), dtype float32, input of Conv1D.

        Returns:
            Variable: shape(B, C_out, T), dtype float32, output of Conv1D.
        """
        # it ensures that ouput time steps == input time steps
        time_steps = x.shape[-1]
        x = super(Conv1DCell, self).forward(x)
        if x.shape[-1] != time_steps:
            x = x[:, :, :time_steps]
        return x

    @property
    def receptive_field(self):
        return self._receptive_field

    def start_sequence(self):
        """Prepare the Conv1DCell to generate a new sequence, this method should be called before calling add_input multiple times.

        WARNING: 
            This method accesses `self.weight` directly. If a `Conv1DCell` object is wrapped in a `WeightNormWrapper`, make sure this method is called only after the `WeightNormWrapper`'s hook is called. 
            `WeightNormWrapper` removes the wrapped layer's `weight`, add has a `weight_v` and `weight_g` to re-compute the wrapped layer's weight as $weight = weight_g * weight_v / ||weight_v||$. (Recomputing the `weight` is a hook before calling the wrapped layer's `forward` method.)
            Whenever a `WeightNormWrapper`'s `forward` method is called, the wrapped layer's weight is updated. But when loading from a checkpoint, `weight_v` and `weight_g` are updated but the wrapped layer's weight is not, since it is no longer a `Parameter`. You should manually call `remove_weight_norm` or `hook` to re-compute the wrapped layer's weight before calling this method if you don't call `forward` first.
            So when loading a model which uses `Conv1DCell` objects wrapped in `WeightNormWrapper`s, remember to call `remove_weight_norm` for all `WeightNormWrapper`s before synthesizing. Also, removing weight norm speeds up computation.
        """
        if not self.causal:
            raise ValueError(
                "Only causal conv1d shell should use start sequence")
        if self.receptive_field == 1:
            raise ValueError(
                "Convolution block with receptive field = 1 does not need"
                " to be implemented as a Conv1DCell. Conv1D suffices")
        self._buffer = None
        self._reshaped_weight = F.reshape(self.weight, (self._num_filters, -1))

    def add_input(self, x_t):
        """This method works similarily with forward but in a `step-in-step-out` fashion.

        Args:
            x (Variable): shape(B, C_in, T=1), dtype float32, input of Conv1D.

        Returns:
            Variable: shape(B, C_out, T=1), dtype float32, output of Conv1D.
        """
        batch_size, c_in, _ = x_t.shape
        if self._buffer is None:
            self._buffer = F.zeros(
                (batch_size, c_in, self.receptive_field), dtype=x_t.dtype)
        self._buffer = F.concat([self._buffer[:, :, 1:], x_t], -1)
        if self._dilation[1] > 1:
            input = F.strided_slice(
                self._buffer,
                axes=[2],
                starts=[0],
                ends=[self.receptive_field],
                strides=[self._dilation[1]])
        else:
            input = self._buffer
        input = F.reshape(input, (batch_size, -1))
        y_t = F.matmul(input, self._reshaped_weight, transpose_y=True)
        y_t = y_t + self.bias
        y_t = F.unsqueeze(y_t, [-1])
        return y_t
