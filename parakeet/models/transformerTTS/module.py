import math
from parakeet.g2p.text.symbols import symbols
import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from layers import Conv1D, Pool1D, DynamicGRU
import numpy as np

class FC(dg.Layer):
    def __init__(self, name_scope, in_features, out_features, is_bias=True, dtype="float32", gain=1):
        super(FC, self).__init__(name_scope)
        self.in_features = in_features
        self.out_features = out_features
        self.is_bias = is_bias
        self.dtype = dtype
        self.gain = gain

        self.weight = self.create_parameter(fluid.ParamAttr(name='weight'), shape=(in_features, out_features), 
                                            dtype=dtype,
                                            default_initializer = fluid.initializer.XavierInitializer())
        #self.weight = gain * self.weight
        # mind the implicit conversion to ParamAttr for many cases                                    
        if is_bias is not False:
            k = math.sqrt(1 / in_features)
            self.bias = self.create_parameter(fluid.ParamAttr(name='bias'), shape=(out_features, ),
                                              is_bias=True,
                                              dtype=dtype,
                                              default_initializer = fluid.initializer.Uniform(low=-k, high=k))

        # 默认初始化权重使用 Xavier 的方法，偏置使用均匀分布，范围是(-\sqrt{k},/sqrt{k}),k=1/infeature
    
    def forward(self, x):
        x = fluid.layers.matmul(x, self.weight)
        if hasattr(self, "bias"):
            x = fluid.layers.elementwise_add(x, self.bias)
        return x

class Conv(dg.Layer):
    def __init__(self, name_scope, in_channels, out_channels, filter_size=1,
                padding=0, dilation=1, stride=1, use_cudnn=True, 
                data_format="NCT", is_bias=True, gain=1):
        super(Conv, self).__init__(name_scope)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.use_cudnn = use_cudnn
        self.data_format = data_format
        self.is_bias = is_bias
        self.gain = gain

        self.weight_attr = fluid.ParamAttr(name='weight', initializer=fluid.initializer.XavierInitializer())
        self.bias_attr = None
        if is_bias is not False:
            k = math.sqrt(1 / in_channels)
            self.bias_attr = fluid.ParamAttr(name='bias', initializer=fluid.initializer.Uniform(low=-k, high=k)) 
        
        self.conv = Conv1D( self.full_name(),
                            in_channels = in_channels,
                            num_filters = out_channels,
                            filter_size = filter_size,
                            padding = padding,
                            dilation = dilation,
                            stride = stride,
                            param_attr = self.weight_attr,
                            bias_attr = self.bias_attr,
                            use_cudnn = use_cudnn,
                            data_format = data_format)

    def forward(self, x):
        x = self.conv(x)
        return x

class EncoderPrenet(dg.Layer):
    def __init__(self, name_scope, embedding_size, num_hidden, use_cudnn=True):
        super(EncoderPrenet, self).__init__(name_scope)
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden
        self.use_cudnn = use_cudnn
        self.embedding = dg.Embedding(self.full_name(), 
                                        size = [len(symbols), embedding_size], 
                                        param_attr = fluid.ParamAttr(name='weight'),
                                        padding_idx = None)
        self.conv1 = Conv(self.full_name(),
                            in_channels = embedding_size, 
                            out_channels = num_hidden, 
                            filter_size = 5,
                            padding = int(np.floor(5/2)),
                            use_cudnn = use_cudnn,
                            data_format = "NCT",
                            gain = math.sqrt(2))
        self.conv2 = Conv(self.full_name(),
                            in_channels = num_hidden, 
                            out_channels = num_hidden, 
                            filter_size = 5,
                            padding = int(np.floor(5/2)),
                            use_cudnn = use_cudnn,
                            data_format = "NCT",
                            gain = math.sqrt(2))
        self.conv3 = Conv(self.full_name(),
                            in_channels = num_hidden, 
                            out_channels = num_hidden, 
                            filter_size = 5,
                            padding = int(np.floor(5/2)),
                            use_cudnn = use_cudnn,
                            data_format = "NCT",
                            gain = math.sqrt(2))
        
        self.batch_norm1 = dg.BatchNorm(self.full_name(), num_hidden, 
                            param_attr = fluid.ParamAttr(name='weight'), 
                            bias_attr = fluid.ParamAttr(name='bias'),
                            moving_mean_name =  'moving_mean',
                            moving_variance_name = 'moving_var',
                            data_layout='NCHW')
        self.batch_norm2 = dg.BatchNorm(self.full_name(), num_hidden, 
                            param_attr = fluid.ParamAttr(name='weight'), 
                            bias_attr = fluid.ParamAttr(name='bias'),
                            moving_mean_name =  'moving_mean',
                            moving_variance_name = 'moving_var',
                            data_layout='NCHW')
        self.batch_norm3 = dg.BatchNorm(self.full_name(), num_hidden, 
                            param_attr = fluid.ParamAttr(name='weight'), 
                            bias_attr = fluid.ParamAttr(name='bias'),
                            moving_mean_name =  'moving_mean',
                            moving_variance_name = 'moving_var',
                            data_layout='NCHW')

        self.projection = FC(self.full_name(), num_hidden, num_hidden)

    def forward(self, x):
        x = self.embedding(fluid.layers.unsqueeze(x, axes=[-1])) #(batch_size, seq_len, embending_size)
        x = layers.transpose(x,[0,2,1])
        x = layers.dropout(layers.relu(self.batch_norm1(self.conv1(x))), 0.2)
        x = layers.dropout(layers.relu(self.batch_norm2(self.conv2(x))), 0.2)
        x = layers.dropout(layers.relu(self.batch_norm3(self.conv3(x))), 0.2)
        x = layers.transpose(x,[0,2,1]) #(N,T,C)
        x = self.projection(x)
        return x

class FFN(dg.Layer):
    def __init__(self, name_scope, num_hidden, use_cudnn=True):
        super(FFN, self).__init__(name_scope)
        self.num_hidden = num_hidden
        self.use_cudnn = use_cudnn
        self.w_1 = Conv(self.full_name(),
                          in_channels = num_hidden, 
                          out_channels = num_hidden * 4, 
                          filter_size = 1,
                          use_cudnn = use_cudnn,
                          data_format = "NCT",
                          gain = math.sqrt(2))
        self.w_2 = Conv(self.full_name(),
                          in_channels = num_hidden * 4,
                          out_channels = num_hidden,
                          filter_size = 1,
                          use_cudnn = use_cudnn,
                          data_format = "NCT",
                          gain = math.sqrt(2))
        self.layer_norm = dg.LayerNorm(self.full_name(), begin_norm_axis=2)

    def forward(self, input):
        #FFN Networt
        x = layers.transpose(input, [0,2,1])
        x = self.w_2(layers.relu(self.w_1(x)))
        x = layers.transpose(x,[0,2,1])
        
        # dropout
        # x = layers.dropout(x, 0.1)
        # not sure where dropout should be placed, in paper should before residual, 
        # but the diagonal alignment did not appear correctly in the attention plot.

        # residual connection
        x = x + input
        

        #layer normalization
        x = self.layer_norm(x)

        return x

class DecoderPrenet(dg.Layer):
    def __init__(self, name_scope, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(DecoderPrenet, self).__init__(name_scope)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        self.fc1 = FC(self.full_name(), input_size, hidden_size) #in pytorch this gian=1
        self.fc2 = FC(self.full_name(), hidden_size, output_size)

    def forward(self, x):
        x = layers.dropout(layers.relu(self.fc1(x)), self.dropout_rate)
        x = layers.dropout(layers.relu(self.fc2(x)), self.dropout_rate)
        return x

class ScaledDotProductAttention(dg.Layer):
    def __init__(self, name_scope, d_key):
        super(ScaledDotProductAttention, self).__init__(name_scope)

        self.d_key = d_key
    
    # please attention this mask is diff from pytorch
    def forward(self, key, value, query, mask=None, query_mask=None):
        # Compute attention score
        attention = layers.matmul(query, key, transpose_y=True) #transpose the last dim in y
        attention = attention / math.sqrt(self.d_key)

        # Mask key to ignore padding
        if mask is not None:
            attention = attention * mask
            mask = (mask == 0).astype(float) * (-2 ** 32 + 1)
            attention = attention + mask

        attention = layers.softmax(attention)
        # Mask query to ignore padding
        # Not sure how to work
        if query_mask is not None:
            attention = attention * query_mask
        
        result = layers.matmul(attention, value)
        return result, attention

class MultiheadAttention(dg.Layer):
    def __init__(self, name_scope, num_hidden, num_head=4):
        super(MultiheadAttention, self).__init__(name_scope)
        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // num_head
        self.num_head = num_head

        self.key = FC(self.full_name(), num_hidden, num_hidden, is_bias=False)
        self.value = FC(self.full_name(), num_hidden, num_hidden, is_bias=False)
        self.query = FC(self.full_name(), num_hidden, num_hidden, is_bias=False)

        self.scal_attn = ScaledDotProductAttention(self.full_name(), self.num_hidden_per_attn)

        self.fc = FC(self.full_name(), num_hidden * 2, num_hidden)

        self.layer_norm = dg.LayerNorm(self.full_name(), begin_norm_axis=2)

    def forward(self, key, value, query_input, mask=None, query_mask=None):
        batch_size = key.shape[0]
        seq_len_key = key.shape[1]
        seq_len_query = query_input.shape[1]

        # repeat masks h times
        if query_mask is not None:
            query_mask = layers.unsqueeze(query_mask, axes=[-1])
            query_mask = layers.expand(query_mask, [self.num_head, 1, seq_len_key])
        if mask is not None:
            mask = layers.expand(mask, (self.num_head, 1, 1))
        
        # Make multihead attention
        # key & value.shape = (batch_size, seq_len, feature)(feature = num_head * num_hidden_per_attn)
        key = layers.reshape(self.key(key), [batch_size, seq_len_key, self.num_head, self.num_hidden_per_attn])
        value = layers.reshape(self.value(value), [batch_size, seq_len_key, self.num_head, self.num_hidden_per_attn])
        query = layers.reshape(self.query(query_input), [batch_size, seq_len_query, self.num_head, self.num_hidden_per_attn])

        key = layers.reshape(layers.transpose(key, [2, 0, 1, 3]), [-1, seq_len_key, self.num_hidden_per_attn])
        value = layers.reshape(layers.transpose(value, [2, 0, 1, 3]), [-1, seq_len_key, self.num_hidden_per_attn])
        query = layers.reshape(layers.transpose(query, [2, 0, 1, 3]), [-1, seq_len_query, self.num_hidden_per_attn])
        
        result, attention = self.scal_attn(key, value, query, mask=mask, query_mask=query_mask)
        
        # concat all multihead result
        result = layers.reshape(result, [self.num_head, batch_size, seq_len_query, self.num_hidden_per_attn])
        result = layers.reshape(layers.transpose(result, [1,2,0,3]),[batch_size, seq_len_query, -1])
        #print(result.().shape)
        # concat result with input
        result = layers.concat([query_input, result], axis=-1)
        
        result = self.fc(result)
        result = result + query_input
        
        result = self.layer_norm(result)
        return result, attention

class PostConvNet(dg.Layer):
    def __init__(self, name_scope, config):
        super(PostConvNet, self).__init__(name_scope)
        
        num_hidden = config.network.hidden_size
        self.num_hidden = num_hidden
        self.conv1 = Conv(self.full_name(),
                            in_channels = config.audio.num_mels * config.audio.outputs_per_step,
                            out_channels = num_hidden,
                            filter_size = 5,
                            padding = 4,
                            use_cudnn = config.use_gpu,
                            data_format = "NCT",
                            gain = 5 / 3)
        self.conv_list = [Conv(self.full_name(),
                            in_channels = num_hidden,
                            out_channels = num_hidden,
                            filter_size = 5,
                            padding = 4,
                            use_cudnn = config.use_gpu,
                            data_format = "NCT",
                            gain = 5 / 3) for _ in range(3)]
        for i, layer in enumerate(self.conv_list):
            self.add_sublayer("conv_list_{}".format(i), layer)
        self.conv5 = Conv(self.full_name(),
                            in_channels = num_hidden,
                            out_channels = config.audio.num_mels * config.audio.outputs_per_step,
                            filter_size = 5,
                            padding = 4,
                            use_cudnn = config.use_gpu,
                            data_format = "NCT")

        self.batch_norm_list = [dg.BatchNorm(self.full_name(), num_hidden, 
                            param_attr = fluid.ParamAttr(name='weight'), 
                            bias_attr = fluid.ParamAttr(name='bias'),
                            moving_mean_name =  'moving_mean',
                            moving_variance_name = 'moving_var',
                            data_layout='NCHW') for _ in range(3)]
        for i, layer in enumerate(self.batch_norm_list):
            self.add_sublayer("batch_norm_list_{}".format(i), layer)
        self.batch_norm1 = dg.BatchNorm(self.full_name(), num_hidden, 
                            param_attr = fluid.ParamAttr(name='weight'), 
                            bias_attr = fluid.ParamAttr(name='bias'),
                            moving_mean_name =  'moving_mean',
                            moving_variance_name = 'moving_var',
                            data_layout='NCHW')

    def forward(self, input):
        input = layers.dropout(layers.tanh(self.batch_norm1(self.conv1(input)[:, :, :-4])),0.1)
        for batch_norm, conv in zip(self.batch_norm_list, self.conv_list):
            input = layers.dropout(layers.tanh(batch_norm(conv(input)[:, :, :-4])),0.1)
        input = self.conv5(input)[:, :, :-4]
        return input

class CBHG(dg.Layer):
    def __init__(self, name_scope, config, K=16, projection_size = 256, num_gru_layers=2, 
                 max_pool_kernel_size=2, is_post=False):
        super(CBHG, self).__init__(name_scope)
        """
        :param hidden_size: dimension of hidden unit
        :param K: # of convolution banks
        :param projection_size: dimension of projection unit
        :param num_gru_layers: # of layers of GRUcell
        :param max_pool_kernel_size: max pooling kernel size
        :param is_post: whether post processing or not
        """
        hidden_size = config.network.hidden_size
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.conv_list = []
        self.conv_list.append(Conv(self.full_name(),
                            in_channels = projection_size,
                            out_channels = hidden_size,
                            filter_size = 1,
                            padding = int(np.floor(1/2)),
                            data_format = "NCT"))
        for i in range(2,K+1):
            self.conv_list.append(Conv(self.full_name(),
                            in_channels = hidden_size,
                            out_channels = hidden_size,
                            filter_size = i,
                            padding = int(np.floor(i/2)),
                            data_format = "NCT"))

        for i, layer in enumerate(self.conv_list):
            self.add_sublayer("conv_list_{}".format(i), layer)

        self.batchnorm_list = []
        for i in range(K):
            self.batchnorm_list.append(dg.BatchNorm(self.full_name(), hidden_size, 
                            param_attr = fluid.ParamAttr(name='weight'), 
                            bias_attr = fluid.ParamAttr(name='bias'),
                            moving_mean_name =  'moving_mean',
                            moving_variance_name = 'moving_var',
                            data_layout='NCHW'))

        for i, layer in enumerate(self.batchnorm_list):
            self.add_sublayer("batchnorm_list_{}".format(i), layer)

        conv_outdim = hidden_size * K

        self.conv_projection_1 = Conv(self.full_name(),
                            in_channels = conv_outdim,
                            out_channels = hidden_size,
                            filter_size = 3,
                            padding = int(np.floor(3/2)),
                            data_format = "NCT")

        self.conv_projection_2 = Conv(self.full_name(),
                            in_channels = hidden_size,
                            out_channels = projection_size,
                            filter_size = 3,
                            padding = int(np.floor(3/2)),
                            data_format = "NCT")

        self.batchnorm_proj_1 = dg.BatchNorm(self.full_name(), hidden_size, 
                            param_attr = fluid.ParamAttr(name='weight'), 
                            bias_attr = fluid.ParamAttr(name='bias'),
                            moving_mean_name =  'moving_mean',
                            moving_variance_name = 'moving_var',
                            data_layout='NCHW')
        self.batchnorm_proj_2 = dg.BatchNorm(self.full_name(), projection_size, 
                            param_attr = fluid.ParamAttr(name='weight'), 
                            bias_attr = fluid.ParamAttr(name='bias'),
                            moving_mean_name =  'moving_mean',
                            moving_variance_name = 'moving_var',
                            data_layout='NCHW')
        self.max_pool = Pool1D(self.full_name(), pool_size = max_pool_kernel_size, 
                    pool_type='max', 
                    pool_stride=1, 
                    pool_padding=1,
                    data_format = "NCT")
        self.highway = Highwaynet(self.full_name(), self.projection_size)

        h_0 = np.zeros((config.batch_size, hidden_size // 2), dtype="float32")
        h_0 = dg.to_variable(h_0)
        self.fc_forward1 = FC(self.full_name(), hidden_size, hidden_size // 2 * 3)
        self.fc_reverse1 = FC(self.full_name(), hidden_size, hidden_size // 2 * 3)
        self.gru_forward1 = DynamicGRU(self.full_name(),
                              size = self.hidden_size // 2,
                              param_attr = fluid.ParamAttr(name='weight'), 
                              bias_attr = fluid.ParamAttr(name='bias'),
                              is_reverse = False,
                              origin_mode = True,
                              h_0 = h_0)
        self.gru_reverse1 = DynamicGRU(self.full_name(),
                              size = self.hidden_size // 2,
                              param_attr = fluid.ParamAttr(name='weight'), 
                              bias_attr = fluid.ParamAttr(name='bias'),
                              is_reverse=True,
                              origin_mode=True,
                              h_0 = h_0)

        self.fc_forward2 = FC(self.full_name(), hidden_size, hidden_size // 2 * 3)
        self.fc_reverse2 = FC(self.full_name(), hidden_size, hidden_size // 2 * 3)
        self.gru_forward2 = DynamicGRU(self.full_name(),
                              size = self.hidden_size // 2,
                              param_attr = fluid.ParamAttr(name='weight'), 
                              bias_attr = fluid.ParamAttr(name='bias'),
                              is_reverse = False,
                              origin_mode = True,
                              h_0 = h_0)
        self.gru_reverse2 = DynamicGRU(self.full_name(),
                              size = self.hidden_size // 2,
                              param_attr = fluid.ParamAttr(name='weight'), 
                              bias_attr = fluid.ParamAttr(name='bias'),
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
    def __init__(self, name_scope, num_units, num_layers=4):
        super(Highwaynet, self).__init__(name_scope)
        self.num_units = num_units
        self.num_layers = num_layers

        self.gates = []
        self.linears = []

        for i in range(num_layers):
            self.linears.append(FC(self.full_name(), num_units, num_units))
            self.gates.append(FC(self.full_name(), num_units, num_units))
        
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




                
        
