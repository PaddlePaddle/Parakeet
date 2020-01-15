import math
from parakeet.g2p.text.symbols import symbols
import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from parakeet.modules.layers import Conv, Pool1D, Linear
from parakeet.modules.dynamicGRU import DynamicGRU
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

class CBHG(dg.Layer):
    def __init__(self, hidden_size, batch_size, K=16, projection_size = 256, num_gru_layers=2, 
                 max_pool_kernel_size=2, is_post=False):
        super(CBHG, self).__init__()
        """
        :param hidden_size: dimension of hidden unit
        :param batch_size: batch size
        :param K: # of convolution banks
        :param projection_size: dimension of projection unit
        :param num_gru_layers: # of layers of GRUcell
        :param max_pool_kernel_size: max pooling kernel size
        :param is_post: whether post processing or not
        """
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.conv_list = []
        self.conv_list.append(Conv(in_channels = projection_size,
                            out_channels = hidden_size,
                            filter_size = 1,
                            padding = int(np.floor(1/2)),
                            data_format = "NCT"))
        for i in range(2,K+1):
            self.conv_list.append(Conv(in_channels = hidden_size,
                            out_channels = hidden_size,
                            filter_size = i,
                            padding = int(np.floor(i/2)),
                            data_format = "NCT"))

        for i, layer in enumerate(self.conv_list):
            self.add_sublayer("conv_list_{}".format(i), layer)

        self.batchnorm_list = []
        for i in range(K):
            self.batchnorm_list.append(dg.BatchNorm(hidden_size, 
                            data_layout='NCHW'))

        for i, layer in enumerate(self.batchnorm_list):
            self.add_sublayer("batchnorm_list_{}".format(i), layer)

        conv_outdim = hidden_size * K

        self.conv_projection_1 = Conv(in_channels = conv_outdim,
                            out_channels = hidden_size,
                            filter_size = 3,
                            padding = int(np.floor(3/2)),
                            data_format = "NCT")

        self.conv_projection_2 = Conv(in_channels = hidden_size,
                            out_channels = projection_size,
                            filter_size = 3,
                            padding = int(np.floor(3/2)),
                            data_format = "NCT")

        self.batchnorm_proj_1 = dg.BatchNorm(hidden_size, 
                            data_layout='NCHW')
        self.batchnorm_proj_2 = dg.BatchNorm(projection_size, 
                            data_layout='NCHW')
        self.max_pool = Pool1D(pool_size = max_pool_kernel_size, 
                    pool_type='max', 
                    pool_stride=1, 
                    pool_padding=1,
                    data_format = "NCT")
        self.highway = Highwaynet(self.projection_size)

        h_0 = np.zeros((batch_size, hidden_size // 2), dtype="float32")
        h_0 = dg.to_variable(h_0)
        self.fc_forward1 = Linear(hidden_size, hidden_size // 2 * 3)
        self.fc_reverse1 = Linear(hidden_size, hidden_size // 2 * 3)
        self.gru_forward1 = DynamicGRU(size = self.hidden_size // 2,
                              is_reverse = False,
                              origin_mode = True,
                              h_0 = h_0)
        self.gru_reverse1 = DynamicGRU(size = self.hidden_size // 2,
                              is_reverse=True,
                              origin_mode=True,
                              h_0 = h_0)

        self.fc_forward2 = Linear(hidden_size, hidden_size // 2 * 3)
        self.fc_reverse2 = Linear(hidden_size, hidden_size // 2 * 3)
        self.gru_forward2 = DynamicGRU(size = self.hidden_size // 2,
                              is_reverse = False,
                              origin_mode = True,
                              h_0 = h_0)
        self.gru_reverse2 = DynamicGRU(size = self.hidden_size // 2,
                              is_reverse=True,
                              origin_mode=True,
                              h_0 = h_0)

    def _conv_fit_dim(self, x, filter_size=3):
        if filter_size % 2 == 0:
            return x[:,:,:-1]
        else:
            return x 

    def forward(self, input_):
        # input_.shape = [N, C, T]

        conv_list = []
        conv_input = input_
        
        for i, (conv, batchnorm) in enumerate(zip(self.conv_list, self.batchnorm_list)):
            conv_input = self._conv_fit_dim(conv(conv_input), i+1)
            conv_input = layers.relu(batchnorm(conv_input))
            conv_list.append(conv_input)
        
        conv_cat = layers.concat(conv_list, axis=1)
        conv_pool = self.max_pool(conv_cat)[:,:,:-1]
        
        
        conv_proj = layers.relu(self.batchnorm_proj_1(self._conv_fit_dim(self.conv_projection_1(conv_pool))))
        conv_proj = self.batchnorm_proj_2(self._conv_fit_dim(self.conv_projection_2(conv_proj))) + input_
        
        # conv_proj.shape = [N, C, T]
        highway = layers.transpose(conv_proj, [0,2,1])
        highway = self.highway(highway)

        # highway.shape = [N, T, C]
        fc_forward = self.fc_forward1(highway)
        fc_reverse = self.fc_reverse1(highway)
        out_forward = self.gru_forward1(fc_forward)
        out_reverse = self.gru_reverse1(fc_reverse)
        out = layers.concat([out_forward, out_reverse], axis=-1)
        fc_forward = self.fc_forward2(out)
        fc_reverse = self.fc_reverse2(out)
        out_forward = self.gru_forward2(fc_forward)
        out_reverse = self.gru_reverse2(fc_reverse)
        out = layers.concat([out_forward, out_reverse], axis=-1)
        out = layers.transpose(out, [0,2,1])
        return out

class Highwaynet(dg.Layer):
    def __init__(self, num_units, num_layers=4):
        super(Highwaynet, self).__init__()
        self.num_units = num_units
        self.num_layers = num_layers

        self.gates = []
        self.linears = []

        for i in range(num_layers):
            self.linears.append(Linear(num_units, num_units))
            self.gates.append(Linear(num_units, num_units))
        
        for i, (linear, gate) in enumerate(zip(self.linears,self.gates)):
            self.add_sublayer("linears_{}".format(i), linear)
            self.add_sublayer("gates_{}".format(i), gate)

    def forward(self, input_):
        out = input_

        for linear, gate in zip(self.linears, self.gates):
            h = fluid.layers.relu(linear(out))
            t_ = fluid.layers.sigmoid(gate(out))

            c = 1 - t_
            out  = h * t_ + out  * c
            
        return out




                
        
