import math
from parakeet.g2p.text.symbols import symbols
import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from parakeet.modules.layers import Conv, Linear
import numpy as np


class EncoderPrenet(dg.Layer):
    def __init__(self, embedding_size, num_hidden, use_cudnn=True):
        super(EncoderPrenet, self).__init__()
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden
        self.use_cudnn = use_cudnn
        self.embedding = dg.Embedding( size = [len(symbols), embedding_size],
                                        padding_idx = None)
        self.conv_list = []
        self.conv_list.append(Conv(in_channels = embedding_size, 
                            out_channels = num_hidden, 
                            filter_size = 5,
                            padding = int(np.floor(5/2)),
                            use_cudnn = use_cudnn,
                            data_format = "NCT"))
        for _ in range(2):
            self.conv_list.append(Conv(in_channels = num_hidden, 
                                out_channels = num_hidden, 
                                filter_size = 5,
                                padding = int(np.floor(5/2)),
                                use_cudnn = use_cudnn,
                                data_format = "NCT"))

        for i, layer in enumerate(self.conv_list):
            self.add_sublayer("conv_list_{}".format(i), layer)

        self.batch_norm_list = [dg.BatchNorm(num_hidden, 
                            data_layout='NCHW') for _ in range(3)]

        for i, layer in enumerate(self.batch_norm_list):
            self.add_sublayer("batch_norm_list_{}".format(i), layer)

        self.projection = Linear(num_hidden, num_hidden)

    def forward(self, x):
        x = self.embedding(x) #(batch_size, seq_len, embending_size)
        x = layers.transpose(x,[0,2,1])
        for batch_norm, conv in zip(self.batch_norm_list, self.conv_list):
            x = layers.dropout(layers.relu(batch_norm(conv(x))), 0.2)
        x = layers.transpose(x,[0,2,1]) #(N,T,C)
        x = self.projection(x)

        return x