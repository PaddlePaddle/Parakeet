import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers
import paddle.fluid as fluid
import math
from parakeet.modules.layers import Conv


class PositionwiseFeedForward(dg.Layer):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, num_hidden, filter_size, padding=0, use_cudnn=True, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.num_hidden = num_hidden
        self.use_cudnn = use_cudnn
        self.dropout = dropout

        self.w_1 = Conv(in_channels = d_in, 
                        out_channels = num_hidden, 
                        filter_size = filter_size,
                        padding=padding,
                        use_cudnn = use_cudnn,
                        data_format = "NTC")
        
        self.w_2 = Conv(in_channels = num_hidden,
                        out_channels = d_in,
                        filter_size = filter_size,
                        padding=padding,
                        use_cudnn = use_cudnn,
                        data_format = "NTC")
        self.layer_norm = dg.LayerNorm(d_in)

    def forward(self, input):
        """
        Feed Forward Network.
        
        Args:
            input (Variable): Shape(B, T, C), dtype: float32. The input value.
        Returns:
            output (Variable), Shape(B, T, C), the result after FFN.
        """
        #FFN Networt
        x = self.w_2(layers.relu(self.w_1(input)))
        
        # dropout
        x = layers.dropout(x, self.dropout)

        # residual connection
        x = x + input
        
        #layer normalization
        output = self.layer_norm(x)

        return output