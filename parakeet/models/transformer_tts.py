import math
import paddle
from paddle import nn
from paddle.nn import functional as F

from parakeet.modules.attention import _split_heads, _concat_heads, drop_head, scaled_dot_product_attention
from parakeet.modules.transformer import PositionwiseFFN
from parakeet.modules import masking
from parakeet.modules.cbhg import Conv1dBatchNorm
from parakeet.modules import positional_encoding as pe

__all__ = ["TransformerTTS"]

# Transformer TTS's own implementation of transformer
class MultiheadAttention(nn.Layer):
    """
    Multihead scaled dot product attention with drop head. See 
    [Scheduled DropHead: A Regularization Method for Transformer Models](https://arxiv.org/abs/2004.13342) 
    for details.
    
    Another deviation is that it concats the input query and context vector before
    applying the output projection.
    """
    def __init__(self, model_dim, num_heads, k_dim=None, v_dim=None):
        """
        Args:
            model_dim (int): the feature size of query.
            num_heads (int): the number of attention heads.
            k_dim (int, optional): feature size of the key of each scaled dot 
                product attention. If not provided, it is set to 
                model_dim / num_heads. Defaults to None.
            v_dim (int, optional): feature size of the key of each scaled dot 
                product attention. If not provided, it is set to 
                model_dim / num_heads. Defaults to None.

        Raises:
            ValueError: if model_dim is not divisible by num_heads
        """
        super(MultiheadAttention, self).__init__()
        if model_dim % num_heads !=0:
            raise ValueError("model_dim must be divisible by num_heads")
        depth = model_dim // num_heads
        k_dim = k_dim or depth
        v_dim = v_dim or depth
        self.affine_q = nn.Linear(model_dim, num_heads * k_dim)
        self.affine_k = nn.Linear(model_dim, num_heads * k_dim)
        self.affine_v = nn.Linear(model_dim, num_heads * v_dim)
        self.affine_o = nn.Linear(model_dim + num_heads * v_dim, model_dim)
        
        self.num_heads = num_heads
        self.model_dim = model_dim
    
    def forward(self, q, k, v, mask, drop_n_heads=0):
        """
        Compute context vector and attention weights.
        
        Args:
            q (Tensor): shape(batch_size, time_steps_q, model_dim), the queries.
            k (Tensor): shape(batch_size, time_steps_k, model_dim), the keys.
            v (Tensor): shape(batch_size, time_steps_k, model_dim), the values.
            mask (Tensor): shape(batch_size, times_steps_q, time_steps_k) or 
                broadcastable shape, dtype: float32 or float64, the mask.

        Returns:
            (out, attention_weights)
            out (Tensor), shape(batch_size, time_steps_q, model_dim), the context vector.
            attention_weights (Tensor): shape(batch_size, times_steps_q, time_steps_k), the attention weights.
        """
        q_in = q
        q = _split_heads(self.affine_q(q), self.num_heads) # (B, h, T, C)
        k = _split_heads(self.affine_k(k), self.num_heads)
        v = _split_heads(self.affine_v(v), self.num_heads)
        mask = paddle.unsqueeze(mask, 1) # unsqueeze for the h dim
        
        context_vectors, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        context_vectors = drop_head(context_vectors, drop_n_heads, self.training)
        context_vectors = _concat_heads(context_vectors) # (B, T, h*C)
        
        concat_feature = paddle.concat([q_in, context_vectors], -1)
        out = self.affine_o(concat_feature)
        return out, attention_weights


class TransformerEncoderLayer(nn.Layer):
    """
    Transformer encoder layer.
    """
    def __init__(self, d_model, n_heads, d_ffn, dropout=0.):
        """
        Args:
            d_model (int): the feature size of the input, and the output.
            n_heads (int): the number of heads in the internal MultiHeadAttention layer.
            d_ffn (int): the hidden size of the internal PositionwiseFFN.
            dropout (float, optional): the probability of the dropout in 
                MultiHeadAttention and PositionwiseFFN. Defaults to 0.
        """
        super(TransformerEncoderLayer, self).__init__()
        self.self_mha = MultiheadAttention(d_model, n_heads)
        self.layer_norm1 = nn.LayerNorm([d_model], epsilon=1e-6)
        
        self.ffn = PositionwiseFFN(d_model, d_ffn, dropout)
        self.layer_norm2 = nn.LayerNorm([d_model], epsilon=1e-6)
    
    def forward(self, x, mask):
        """
        Args:
            x (Tensor): shape(batch_size, time_steps, d_model), the decoder input.
            mask (Tensor): shape(batch_size, time_steps), the padding mask.
        
        Returns:
            (x, attn_weights)
            x (Tensor): shape(batch_size, time_steps, d_model), the decoded.
            attn_weights (Tensor), shape(batch_size, n_heads, time_steps, time_steps), self attention.
        """
        # pre norm
        x_in = x
        x = self.layer_norm1(x)
        context_vector, attn_weights = self.self_mha(x, x, x, paddle.unsqueeze(mask, 1))
        x = x_in + context_vector # here, the order can be tuned
        
        # pre norm
        x = x + self.ffn(self.layer_norm2(x))
        return x, attn_weights


class TransformerDecoderLayer(nn.Layer):
    """
    Transformer decoder layer.
    """
    def __init__(self, d_model, n_heads, d_ffn, dropout=0.):
        """
        Args:
            d_model (int): the feature size of the input, and the output.
            n_heads (int): the number of heads in the internal MultiHeadAttention layer.
            d_ffn (int): the hidden size of the internal PositionwiseFFN.
            dropout (float, optional): the probability of the dropout in 
                MultiHeadAttention and PositionwiseFFN. Defaults to 0.
        """
        super(TransformerDecoderLayer, self).__init__()
        self.self_mha = MultiheadAttention(d_model, n_heads)
        self.layer_norm1 = nn.LayerNorm([d_model], epsilon=1e-6)
        
        self.cross_mha = MultiheadAttention(d_model, n_heads)
        self.layer_norm2 = nn.LayerNorm([d_model], epsilon=1e-6)
        
        self.ffn = PositionwiseFFN(d_model, d_ffn, dropout)
        self.layer_norm3 = nn.LayerNorm([d_model], epsilon=1e-6)
    
    def forward(self, q, k, v, encoder_mask, decoder_mask):
        """
        Args:
            q (Tensor): shape(batch_size, time_steps_q, d_model), the decoder input.
            k (Tensor): shape(batch_size, time_steps_k, d_model), keys.
            v (Tensor): shape(batch_size, time_steps_k, d_model), values
            encoder_mask (Tensor): shape(batch_size, time_steps_k) encoder padding mask.
            decoder_mask (Tensor): shape(batch_size, time_steps_q, time_steps_q) or broadcastable shape, decoder padding mask.
        
        Returns:
            (q, self_attn_weights, cross_attn_weights)
            q (Tensor): shape(batch_size, time_steps_q, d_model), the decoded.
            self_attn_weights (Tensor), shape(batch_size, n_heads, time_steps_q, time_steps_q), decoder self attention.
            cross_attn_weights (Tensor), shape(batch_size, n_heads, time_steps_q, time_steps_k), decoder-encoder cross attention.
        """        
        # pre norm
        q_in = q
        q = self.layer_norm1(q)
        context_vector, self_attn_weights = self.self_mha(q, q, q, decoder_mask)
        q = q_in + context_vector
        
        # pre norm
        q_in = q
        q = self.layer_norm2(q)
        context_vector, cross_attn_weights = self.cross_mha(q, k, v, paddle.unsqueeze(encoder_mask, 1))
        q = q_in + context_vector
        
        # pre norm
        q = q + self.ffn(self.layer_norm3(q))
        return q, self_attn_weights, cross_attn_weights


class TransformerEncoder(nn.LayerList):
    def __init__(self, d_model, n_heads, d_ffn, n_layers, dropout=0.):
        super(TransformerEncoder, self).__init__()
        for _ in range(n_layers):
            self.append(TransformerEncoderLayer(d_model, n_heads, d_ffn, dropout))

    def forward(self, x, mask):
        attention_weights = []
        for layer in self:
            x, attention_weights_i = layer(x, mask)
            attention_weights.append(attention_weights_i)
        return x, attention_weights


class TransformerDecoder(nn.LayerList):
    def __init__(self, d_model, n_heads, d_ffn, n_layers, dropout=0.):
        super(TransformerDecoder, self).__init__()
        for _ in range(n_layers):
            self.append(TransformerDecoderLayer(d_model, n_heads, d_ffn, dropout))

    def forward(self, q, k, v, encoder_mask, decoder_mask):
        self_attention_weights = []
        cross_attention_weights = []
        for layer in self:
            q, self_attention_weights_i, cross_attention_weights_i = layer(q, k, v, encoder_mask, decoder_mask)
            self_attention_weights.append(self_attention_weights_i)
            cross_attention_weights.append(cross_attention_weights_i)
        return q, self_attention_weights, cross_attention_weights


class MLPPreNet(nn.Layer):
    def __init__(self, d_input, d_hidden, d_output, dropout):
        super(MLPPreNet, self).__init__()
        self.lin1 = nn.Linear(d_input, d_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(d_hidden, d_output)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        # the original code said also use dropout in inference
        return self.dropout2(F.relu(self.lin2(self.dropout1(F.relu(self.lin1(x))))))


class CNNPostNet(nn.Layer):
    def __init__(self, d_input, d_hidden, d_output, kernel_size, n_layers):
        super(CNNPostNet, self).__init__()
        self.convs = nn.LayerList()
        kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, ) 
        padding = (kernel_size[0] - 1, 0)
        for i in range(n_layers):
            c_in = d_input if i == 0 else d_hidden
            c_out = d_output if i == n_layers - 1 else d_hidden
            self.convs.append(
                Conv1dBatchNorm(c_in, c_out, kernel_size, padding=padding))
        self.last_norm = nn.BatchNorm1d(d_output)
    
    def forward(self, x):
        x_in = x
        for layer in self.convs:
            x = paddle.tanh(layer(x))
        x = self.last_norm(x + x_in)
        return x


class TransformerTTS(nn.Layer):
    def __init__(self, vocab_size, padding_idx, d_model, d_mel, n_heads, d_ffn, positional_encoding_scalar,
                 encoder_layers, decoder_layers, d_prenet, d_postnet, postnet_layers, 
                 postnet_kernel_size, max_reduction_factor, dropout):
        super(TransformerTTS, self).__init__()
        self.encoder_prenet = nn.Embedding(vocab_size, d_model, padding_idx)
        self.encoder_pe = pe.positional_encoding(0, 1000, d_model) # it may be extended later
        self.encoder = TransformerEncoder(d_model, n_heads, d_ffn, encoder_layers, dropout)
        
        self.decoder_prenet = MLPPreNet(d_mel, d_prenet, d_model, dropout)
        self.decoder_pe = pe.positional_encoding(0, 1000, d_model) # it may be extended later
        self.decoder = TransformerDecoder(d_model, n_heads, d_ffn, decoder_layers, dropout)
        self.final_proj = nn.Linear(d_model, max_reduction_factor * d_mel)
        self.decoder_postnet = CNNPostNet(d_mel, d_postnet, d_mel, postnet_kernel_size, postnet_layers)
        self.stop_conditioner = nn.Linear(d_mel, 3)
        
        # specs
        self.padding_idx = padding_idx
        self.d_model = d_model
        self.pe_scalar = positional_encoding_scalar
        
        # start and end 
        dtype = paddle.get_default_dtype()
        self.start_vec = paddle.fill_constant([1, d_mel], dtype=dtype, value=0)
        self.end_vec = paddle.fill_constant([1, d_mel], dtype=dtype, value=0)
        self.stop_prob_index = 2
        
        self.max_r = max_reduction_factor
        self.r = max_reduction_factor # set it every call
        
        
    def forward(self, text, mel, stop):
        pass
        
    def encode(self, text):
        T_enc = text.shape[-1]
        embed = self.encoder_prenet(text)
        pe = self.encoder_pe[:T_enc, :] # (T, C)
        x = embed.scale(math.sqrt(self.d_model)) + pe.scale(self.pe_scalar)
        encoder_padding_mask = masking.id_mask(text, self.padding_idx, dtype=x.dtype)
        
        x = F.dropout(x, training=self.training)
        x, attention_weights = self.encoder(x, encoder_padding_mask)
        return x, attention_weights, encoder_padding_mask
    
    def decode(self, encoder_output, input, encoder_padding_mask):
        batch_size, T_dec, mel_dim = input.shape
        no_future_mask = masking.future_mask(T_dec, dtype=input.dtype)
        decoder_padding_mask = masking.feature_mask(input, axis=-1, dtype=input.dtype)
        decoder_mask = masking.combine_mask(decoder_padding_mask.unsqueeze(-1), no_future_mask)
        
        decoder_input = self.decoder_prenet(input)
        decoder_output, _, cross_attention_weights = self.decoder(
            decoder_input, 
            encoder_output, 
            encoder_output, 
            encoder_padding_mask, 
            decoder_mask)

        output_proj = self.final_proj(decoder_output)
        mel_intermediate = paddle.reshape(output_proj, [batch_size, -1, mel_dim])
        stop_logits = self.stop_conditioner(mel_intermediate)
        
        mel_channel_first = paddle.transpose(mel_intermediate, [0, 2, 1])
        mel_output = self.decoder_postnet(mel_channel_first)
        mel_output = paddle.transpose(mel_output, [0, 2, 1])
        return mel_output, mel_intermediate, cross_attention_weights, stop_logits
    
    def predict(self, input, max_length=1000, verbose=True):
        """[summary]

        Args:
            input (Tensor): shape (T), dtype int, input text sequencce.
            max_length (int, optional): max decoder steps. Defaults to 1000.
            verbose (bool, optional): display progress bar. Defaults to True.
        """
        text_input = paddle.unsqueeze(input, 0) # (1, T)
        decoder_input = paddle.unsqueeze(self.start_vec, 0) # (B=1, T, C)
        decoder_output = paddle.unsqueeze(self.start_vec, 0) # (B=1, T, C)
        
        # encoder the text sequence
        encoder_output, encoder_attentions, encoder_padding_mask = self.encode(text_input)
        for _ in range(int(max_length // self.r) + 1):
            mel_output, _, cross_attention_weights, stop_logits = self.decode(
                encoder_output, decoder_input, encoder_padding_mask)
            
            # extract last step and append it to decoder input
            decoder_input = paddle.concat([decoder_input, mel_output[:, -1:, :]], 1)
            # extract last r steps and append it to decoder output
            decoder_output = paddle.concat([decoder_output, mel_output[:, -self.r:, :]], 1)
            
            # stop condition?
            if paddle.argmax(stop_logits[:, -1, :]) == self.stop_prob_index:
                if verbose:
                    print("Hits stop condition.")
                break

        return decoder_output[:, 1:, :], encoder_attentions, cross_attention_weights
        
        
        
        
        