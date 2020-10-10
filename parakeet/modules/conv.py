import paddle
from paddle import nn

class Conv1dCell(nn.Conv1d):
    """
    A subclass of Conv1d layer, which can be used like an RNN cell. It can take 
    step input and return step output. It is done by keeping an internal buffer, 
    when adding a step input, we shift the buffer and return a step output. For 
    single step case, convolution devolves to a linear transformation.
    
    That it can be used as a cell depends on several restrictions:
    1. stride must be 1;
    2. padding must be an asymmetric padding (recpetive_field - 1, 0).
    
    As a result, these arguments are removed form the initializer.
    """
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 weight_attr=None,
                 bias_attr=None):
        _dilation = dilation[0] if isinstance(dilation, (tuple, list)) else dilation
        _kernel_size = kernel_size[0] if isinstance(kernel_size, (tuple, list)) else kernel_size
        self._r = 1 + (_kernel_size - 1) * _dilation
        super(Conv1dCell, self).__init__(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=(self._r - 1, 0), 
            dilation=dilation, 
            weight_attr=weight_attr, 
            bias_attr=bias_attr, 
            data_format="NCL")

    @property
    def receptive_field(self):
        return self._r
    
    def start_sequence(self):
        if self.training:
            raise Exception("only use start_sequence in evaluation")
        self._buffer = None
        self._reshaped_weight = paddle.reshape(
            self.weight, (self._out_channels, -1))
    
    def initialize_buffer(self, x_t):
        batch_size, _ = x_t.shape
        self._buffer = paddle.zeros(
            (batch_size, self._in_channels, self.receptive_field), 
            dtype=x_t.dtype)
    
    def update_buffer(self, x_t):
        self._buffer = paddle.concat(
            [self._buffer[:, :, 1:], paddle.unsqueeze(x_t, -1)], -1)

    def add_input(self, x_t):
        """
        Arguments:
            x_t (Tensor): shape (batch_size, in_channels), step input.
        Rerurns:
            y_t (Tensor): shape (batch_size, out_channels), step output.
        """
        batch_size = x_t.shape[0]
        if self.receptive_field > 1:
            if self._buffer is None:
                self.initialize_buffer(x_t)
            
            # update buffer
            self.update_buffer(x_t)
            if self._dilation[0] > 1:
                input = self._buffer[:, :, ::self._dilation[0]]
            else:
                input = self._buffer
            input = paddle.reshape(input, (batch_size, -1))
        else:
            input = x_t
        y_t = paddle.matmul(input, self._reshaped_weight, transpose_y=True)
        y_t = y_t + self.bias
        return y_t
