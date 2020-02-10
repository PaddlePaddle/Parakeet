import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
from parakeet.modules.layers import Conv1D, Linear
from parakeet.modules.utils import *
from parakeet.modules.multihead_attention import MultiheadAttention
from parakeet.modules.feed_forward import PositionwiseFeedForward
from parakeet.modules.prenet import PreNet
from parakeet.modules.post_convnet import PostConvNet
 
class Decoder(dg.Layer):
    def __init__(self, num_hidden, config, num_head=4):
        super(Decoder, self).__init__()
        self.num_hidden = num_hidden
        param = fluid.ParamAttr()
        self.alpha = self.create_parameter(shape=(1,), attr=param, dtype='float32',
                        default_initializer = fluid.initializer.ConstantInitializer(value=1.0))
        self.pos_inp = get_sinusoid_encoding_table(1024, self.num_hidden, padding_idx=0)
        self.pos_emb = dg.Embedding(size=[1024, num_hidden],
                                 padding_idx=0,
                                 param_attr=fluid.ParamAttr(
                                     initializer=fluid.initializer.NumpyArrayInitializer(self.pos_inp),
                                     trainable=False))
        self.decoder_prenet = PreNet(input_size = config.audio.num_mels, 
                                            hidden_size = num_hidden * 2, 
                                            output_size = num_hidden, 
                                            dropout_rate=0.2)
        self.linear = Linear(num_hidden, num_hidden)

        self.selfattn_layers = [MultiheadAttention(num_hidden, num_hidden//num_head, num_hidden//num_head) for _ in range(3)]
        for i, layer in enumerate(self.selfattn_layers):
            self.add_sublayer("self_attn_{}".format(i), layer)
        self.attn_layers = [MultiheadAttention(num_hidden, num_hidden//num_head, num_hidden//num_head) for _ in range(3)]
        for i, layer in enumerate(self.attn_layers):
            self.add_sublayer("attn_{}".format(i), layer)
        self.ffns = [PositionwiseFeedForward(num_hidden, num_hidden*num_head, filter_size=1) for _ in range(3)]
        for i, layer in enumerate(self.ffns):
            self.add_sublayer("ffns_{}".format(i), layer)
        self.mel_linear = Linear(num_hidden, config.audio.num_mels * config.audio.outputs_per_step)
        self.stop_linear = Linear(num_hidden, 1)

        self.postconvnet = PostConvNet(config.audio.num_mels, config.hidden_size, 
                                       filter_size = 5, padding = 4, num_conv=5, 
                                       outputs_per_step=config.audio.outputs_per_step, 
                                       use_cudnn = config.use_gpu)

    def forward(self, key, value, query, c_mask, positional):

        # get decoder mask with triangular matrix
        
        if fluid.framework._dygraph_tracer()._train_mode:
            m_mask = get_non_pad_mask(positional)
            mask = get_attn_key_pad_mask((positional==0).astype(np.float32), query)
            triu_tensor = dg.to_variable(get_triu_tensor(query.numpy(), query.numpy())).astype(np.float32)
            mask = mask + triu_tensor
            mask = fluid.layers.cast(mask == 0, np.float32)
            
            # (batch_size, decoder_len, encoder_len)
            zero_mask = get_attn_key_pad_mask(layers.squeeze(c_mask,[-1]), query)
        else:
            mask = get_triu_tensor(query.numpy(), query.numpy()).astype(np.float32)
            mask = fluid.layers.cast(dg.to_variable(mask == 0), np.float32)
            m_mask, zero_mask = None, None

        # Decoder pre-network
        query = self.decoder_prenet(query)
        
        # Centered position
        query = self.linear(query)

        # Get position embedding
        positional = self.pos_emb(positional)
        query = positional * self.alpha + query

        #positional dropout
        query = fluid.layers.dropout(query, 0.1)

        # Attention decoder-decoder, encoder-decoder
        selfattn_list = list()
        attn_list = list()
        
        for selfattn, attn, ffn in zip(self.selfattn_layers, self.attn_layers, self.ffns):
            query, attn_dec = selfattn(query, query, query, mask = mask, query_mask = m_mask)
            query, attn_dot = attn(key, value, query, mask = zero_mask, query_mask = m_mask)
            query = ffn(query)
            selfattn_list.append(attn_dec)
            attn_list.append(attn_dot)
        # Mel linear projection
        mel_out = self.mel_linear(query)
        # Post Mel Network
        out = self.postconvnet(mel_out)
        out = mel_out + out
        
        # Stop tokens
        stop_tokens = self.stop_linear(query)
        stop_tokens = layers.squeeze(stop_tokens, [-1])
        stop_tokens = layers.sigmoid(stop_tokens)

        return mel_out, out, attn_list, stop_tokens, selfattn_list
