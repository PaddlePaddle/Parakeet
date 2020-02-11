import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers

class DynamicGRU(dg.Layer):
    def __init__(self,
                 size,
                 param_attr=None,
                 bias_attr=None,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
                 h_0=None,
                 origin_mode=False,
                 init_size=None):
        super(DynamicGRU, self).__init__()
        self.gru_unit = dg.GRUUnit(
            size * 3,
            param_attr=param_attr,
            bias_attr=bias_attr,
            activation=candidate_activation,
            gate_activation=gate_activation,
            origin_mode=origin_mode)
        self.size = size
        self.h_0 = h_0
        self.is_reverse = is_reverse

    def forward(self, inputs):
        """
        Dynamic GRU block.
        
        Args:
            input (Variable): Shape(B, T, C), dtype: float32. The input value.
        Returns:
            output (Variable), Shape(B, T, C), the result compute by GRU.
        """
        hidden = self.h_0
        res = []
        for i in range(inputs.shape[1]):
            if self.is_reverse:
                i = inputs.shape[1] - 1 - i
            input_ = inputs[:, i:i + 1, :]
            input_ = layers.reshape(
                input_, [-1, input_.shape[2]], inplace=False)
            hidden, reset, gate = self.gru_unit(input_, hidden)
            hidden_ = layers.reshape(
                hidden, [-1, 1, hidden.shape[1]], inplace=False)
            res.append(hidden_)
        if self.is_reverse:
            res = res[::-1]
        res = layers.concat(res, axis=1)
        return res

