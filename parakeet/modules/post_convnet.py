import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from parakeet.modules.layers import Conv

class PostConvNet(dg.Layer):
    def __init__(self, 
                 n_mels=80,
                 num_hidden=512,
                 filter_size=5,
                 padding=0,
                 num_conv=5,
                 outputs_per_step=1,
                 use_cudnn=True,
                 dropout=0.1):
        super(PostConvNet, self).__init__()
        
        self.dropout = dropout
        self.conv_list = []
        self.conv_list.append(Conv(in_channels = n_mels * outputs_per_step,
                            out_channels = num_hidden,
                            filter_size = filter_size,
                            padding = padding,
                            use_cudnn = use_cudnn,
                            data_format = "NCT"))

        for _ in range(1, num_conv-1):
            self.conv_list.append(Conv(in_channels = num_hidden,
                                out_channels = num_hidden,
                                filter_size = filter_size,
                                padding = padding,
                                use_cudnn = use_cudnn,
                                data_format = "NCT") )
        
        self.conv_list.append(Conv(in_channels = num_hidden,
                            out_channels = n_mels * outputs_per_step,
                            filter_size = filter_size,
                            padding = padding,
                            use_cudnn = use_cudnn,
                            data_format = "NCT"))

        for i, layer in enumerate(self.conv_list):
            self.add_sublayer("conv_list_{}".format(i), layer)

        self.batch_norm_list = [dg.BatchNorm(num_hidden, 
                            param_attr = fluid.ParamAttr(name='weight'), 
                            bias_attr = fluid.ParamAttr(name='bias'),
                            moving_mean_name =  'moving_mean',
                            moving_variance_name = 'moving_var',
                            data_layout='NCHW') for _ in range(num_conv-1)]
        self.batch_norm_list.append(dg.BatchNorm(n_mels * outputs_per_step, 
                            param_attr = fluid.ParamAttr(name='weight'), 
                            bias_attr = fluid.ParamAttr(name='bias'),
                            moving_mean_name =  'moving_mean',
                            moving_variance_name = 'moving_var',
                            data_layout='NCHW'))
        for i, layer in enumerate(self.batch_norm_list):
            self.add_sublayer("batch_norm_list_{}".format(i), layer)
        

    def forward(self, input):
        """
        Post Conv Net.
        
        Args:
            input (Variable): Shape(B, T, C), dtype: float32. The input value.
        Returns:
            output (Variable), Shape(B, T, C), the result after postconvnet.
        """
        input = layers.transpose(input, [0,2,1])
        len = input.shape[-1]
        for batch_norm, conv in zip(self.batch_norm_list, self.conv_list):
            input = layers.dropout(layers.tanh(batch_norm(conv(input)[:,:,:len])), self.dropout)
        output = layers.transpose(input, [0,2,1])
        return output