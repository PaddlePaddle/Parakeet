import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers

class PreNet(dg.Layer):
    """
    Pre Net before passing through the network
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(PreNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        self.linear1 = dg.Linear(input_size, hidden_size)
        self.linear2 = dg.Linear(hidden_size, output_size)

    def forward(self, x):
        x = layers.dropout(layers.relu(self.linear1(x)), self.dropout_rate)
        x = layers.dropout(layers.relu(self.linear2(x)), self.dropout_rate)
        return x
