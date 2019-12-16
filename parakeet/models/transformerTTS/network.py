from module import *
from utils import get_positional_table, get_sinusoid_encoding_table
import paddle.fluid.dygraph as dg
import paddle.fluid as fluid

class Encoder(dg.Layer):
    def __init__(self, name_scope, embedding_size, num_hidden, config):
        super(Encoder, self).__init__(name_scope)
        self.num_hidden = num_hidden
        param = fluid.ParamAttr(name='alpha')
        self.alpha = self.create_parameter(param, shape=(1, ), dtype='float32',
                        default_initializer = fluid.initializer.ConstantInitializer(value=1.0))
        self.pos_inp = get_sinusoid_encoding_table(1024, self.num_hidden, padding_idx=0)
        self.pos_emb = dg.Embedding(name_scope=self.full_name(),
                                 size=[1024, num_hidden],
                                 padding_idx=0,
                                 param_attr=fluid.ParamAttr(
                                     name='weight',
                                     initializer=fluid.initializer.NumpyArrayInitializer(self.pos_inp),
                                     trainable=False))
        self.encoder_prenet = EncoderPrenet(name_scope = self.full_name(), 
                                            embedding_size = embedding_size, 
                                            num_hidden = num_hidden, 
                                            use_cudnn=config.use_gpu)
        self.layers = [MultiheadAttention(self.full_name(), num_hidden) for _ in range(3)]
        for i, layer in enumerate(self.layers):
            self.add_sublayer("self_attn_{}".format(i), layer)
        self.ffns = [FFN(self.full_name(), num_hidden, use_cudnn = config.use_gpu) for _ in range(3)]
        for i, layer in enumerate(self.ffns):
            self.add_sublayer("ffns_{}".format(i), layer)

    def forward(self, x, positional):
        if fluid.framework._dygraph_tracer()._train_mode:
            query_mask = (positional != 0).astype(float)
            mask = (positional != 0).astype(float)
            mask = fluid.layers.expand(fluid.layers.unsqueeze(mask,[1]), [1,x.shape[1], 1])  
        else:
            query_mask, mask = None, None
        
        # Encoder pre_network
        x = self.encoder_prenet(x) #(N,T,C)
        
        
        # Get positional encoding
        positional = self.pos_emb(fluid.layers.unsqueeze(positional, axes=[-1])) 
        x = positional * self.alpha + x #(N, T, C)
       

        # Positional dropout
        x = layers.dropout(x, 0.1)
        
        # Self attention encoder
        attentions = list()
        for layer, ffn in zip(self.layers, self.ffns):
            x, attention = layer(x, x, x, mask = mask, query_mask = query_mask)
            x = ffn(x)
            attentions.append(attention)

        return x, query_mask, attentions

class Decoder(dg.Layer):
    def __init__(self, name_scope, num_hidden, config):
        super(Decoder, self).__init__(name_scope)
        self.num_hidden = num_hidden
        param = fluid.ParamAttr(name='alpha')
        self.alpha = self.create_parameter(param, shape=(1,), dtype='float32',
                        default_initializer = fluid.initializer.ConstantInitializer(value=1.0))
        self.pos_inp = get_sinusoid_encoding_table(1024, self.num_hidden, padding_idx=0)
        self.pos_emb = dg.Embedding(name_scope=self.full_name(),
                                 size=[1024, num_hidden],
                                 padding_idx=0,
                                 param_attr=fluid.ParamAttr(
                                     name='weight',
                                     initializer=fluid.initializer.NumpyArrayInitializer(self.pos_inp),
                                     trainable=False))
        self.decoder_prenet = DecoderPrenet(self.full_name(),
                                            input_size = config.audio.num_mels, 
                                            hidden_size = num_hidden * 2, 
                                            output_size = num_hidden, 
                                            dropout_rate=0.2)
        self.linear = FC(self.full_name(), num_hidden, num_hidden)

        self.selfattn_layers = [MultiheadAttention(self.full_name(), num_hidden) for _ in range(3)]
        for i, layer in enumerate(self.selfattn_layers):
            self.add_sublayer("self_attn_{}".format(i), layer)
        self.attn_layers = [MultiheadAttention(self.full_name(), num_hidden) for _ in range(3)]
        for i, layer in enumerate(self.attn_layers):
            self.add_sublayer("attn_{}".format(i), layer)
        self.ffns = [FFN(self.full_name(), num_hidden) for _ in range(3)]
        for i, layer in enumerate(self.ffns):
            self.add_sublayer("ffns_{}".format(i), layer)
        self.mel_linear = FC(self.full_name(), num_hidden, config.audio.num_mels * config.audio.outputs_per_step)
        self.stop_linear = FC(self.full_name(), num_hidden, 1, gain = 1)

        self.postconvnet = PostConvNet(self.full_name(), config)

    def forward(self, key, value, query, c_mask, positional):
        batch_size = key.shape[0]
        decoder_len = query.shape[1]

        # get decoder mask with triangular matrix
        
        if fluid.framework._dygraph_tracer()._train_mode:
            #zeros = np.zeros(positional.shape, dtype=np.float32)
            m_mask = (positional != 0).astype(float)
            mask = np.repeat(np.expand_dims(m_mask.numpy() == 0, axis=1), decoder_len, axis=1)
            mask = mask + np.repeat(np.expand_dims(np.triu(np.ones([decoder_len, decoder_len]), 1), axis=0) ,batch_size, axis=0)
            mask = fluid.layers.cast(dg.to_variable(mask == 0), np.float32)
            

            # (batch_size, decoder_len, decoder_len)
            zero_mask = fluid.layers.expand(fluid.layers.unsqueeze((c_mask != 0).astype(float), axes=2), [1,1,decoder_len])
            # (batch_size, decoder_len, seq_len)
            zero_mask = fluid.layers.transpose(zero_mask, [0,2,1])
        
        else:
            mask = np.repeat(np.expand_dims(np.triu(np.ones([decoder_len, decoder_len]), 1), axis=0) ,batch_size, axis=0)
            mask = fluid.layers.cast(dg.to_variable(mask == 0), np.float32)
            m_mask, zero_mask = None, None
        #import pdb; pdb.set_trace()
        # Decoder pre-network
        query = self.decoder_prenet(query)

        # Centered position
        query = self.linear(query)

        # Get position embedding
        positional = self.pos_emb(fluid.layers.unsqueeze(positional, axes=[-1]))
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
        postnet_input = layers.transpose(mel_out, [0,2,1])
        out = self.postconvnet(postnet_input)
        out = postnet_input + out
        out = layers.transpose(out, [0,2,1])
        
        # Stop tokens
        stop_tokens = self.stop_linear(query)

        return mel_out, out, attn_list, stop_tokens, selfattn_list

class Model(dg.Layer):
    def __init__(self, name_scope, config):
        super(Model, self).__init__(name_scope)
        self.encoder = Encoder(self.full_name(), config.network.embedding_size, config.network.hidden_size, config)
        self.decoder = Decoder(self.full_name(), config.network.hidden_size, config)
        self.config = config

    def forward(self, characters, mel_input, pos_text, pos_mel):
        # key (batch_size, seq_len, channel)
        # c_mask (batch_size, seq_len)
        # attns_enc (channel / 2, seq_len, seq_len)
        key, c_mask, attns_enc = self.encoder(characters, pos_text)
        
        # mel_output/postnet_output (batch_size, mel_len, n_mel)
        # attn_probs (128, mel_len, seq_len)
        # stop_preds (batch_size, mel_len, 1)
        # attns_dec (128, mel_len, mel_len)
        mel_output, postnet_output, attn_probs, stop_preds, attns_dec = self.decoder(key, key, mel_input, c_mask, pos_mel)

        return mel_output, postnet_output, attn_probs, stop_preds, attns_enc, attns_dec

class ModelPostNet(dg.Layer):
    """
    CBHG Network (mel -> linear)
    """
    def __init__(self, name_scope, config):
        super(ModelPostNet, self).__init__(name_scope)
        self.pre_proj = Conv(self.full_name(), 
                             in_channels = config.audio.num_mels, 
                             out_channels = config.network.hidden_size,
                             data_format = "NCT")
        self.cbhg = CBHG(self.full_name(), config)
        self.post_proj = Conv(self.full_name(), 
                             in_channels = config.audio.num_mels, 
                             out_channels = (config.audio.n_fft // 2) + 1,
                             data_format = "NCT")

    def forward(self, mel):
        mel = layers.transpose(mel, [0,2,1])
        mel = self.pre_proj(mel)
        mel = self.cbhg(mel)
        mag_pred = self.post_proj(mel)
        mag_pred = layers.transpose(mag_pred, [0,2,1])
        return mag_pred

    


            
            
