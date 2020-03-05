import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
from parakeet.models.transformer_tts.utils import *
from parakeet.modules.multihead_attention import MultiheadAttention
from parakeet.modules.ffn import PositionwiseFeedForward
from parakeet.models.transformer_tts.encoderprenet import EncoderPrenet

class Encoder(dg.Layer):
    def __init__(self, embedding_size, num_hidden, num_head=4):
        super(Encoder, self).__init__()
        self.num_hidden = num_hidden
        self.num_head = num_head
        param = fluid.ParamAttr(initializer=fluid.initializer.Constant(value=1.0))
        self.alpha = self.create_parameter(shape=(1, ), attr=param, dtype='float32')
        self.pos_inp = get_sinusoid_encoding_table(1024, self.num_hidden, padding_idx=0)
        self.pos_emb = dg.Embedding(size=[1024, num_hidden],
                                 param_attr=fluid.ParamAttr(
                                     initializer=fluid.initializer.NumpyArrayInitializer(self.pos_inp),
                                     trainable=False))
        self.encoder_prenet = EncoderPrenet(embedding_size = embedding_size, 
                                            num_hidden = num_hidden, 
                                            use_cudnn=True)
        self.layers = [MultiheadAttention(num_hidden, num_hidden//num_head, num_hidden//num_head) for _ in range(3)]
        for i, layer in enumerate(self.layers):
            self.add_sublayer("self_attn_{}".format(i), layer)
        self.ffns = [PositionwiseFeedForward(num_hidden, num_hidden*num_head, filter_size=1, use_cudnn=True) for _ in range(3)]
        for i, layer in enumerate(self.ffns):
            self.add_sublayer("ffns_{}".format(i), layer)

    def forward(self, x, positional, mask=None, query_mask=None):
        
        if fluid.framework._dygraph_tracer()._train_mode:
            seq_len_key = x.shape[1]
            query_mask = layers.expand(query_mask, [self.num_head, 1, seq_len_key])
            mask = layers.expand(mask, [self.num_head, 1, 1])
        else:
            query_mask, mask = None, None
        
    
        # Encoder pre_network
        x = self.encoder_prenet(x) #(N,T,C)
        
        
        # Get positional encoding
        positional = self.pos_emb(positional) 
        
        x = positional * self.alpha + x #(N, T, C)
       
        
        # Positional dropout
        x = layers.dropout(x, 0.1, dropout_implementation='upscale_in_train')
        
        # Self attention encoder
        attentions = list()
        for layer, ffn in zip(self.layers, self.ffns):
            x, attention = layer(x, x, x, mask = mask, query_mask = query_mask)
            x = ffn(x)
            attentions.append(attention)

        return x, attentions