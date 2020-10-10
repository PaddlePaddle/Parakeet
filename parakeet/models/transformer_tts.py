import math
import paddle
from paddle import nn
from paddle.nn import functional as F

from parakeet.modules.attention import _split_heads, _concat_heads, drop_head, scaled_dot_product_attention
from parakeet.modules.transformer import PositionwiseFFN, combine_mask
from parakeet.modules.cbhg import Conv1dBatchNorm

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
            decoder_mask (Tensor): shape(batch_size, time_steps_q) decoder padding mask.
        
        Returns:
            (q, self_attn_weights, cross_attn_weights)
            q (Tensor): shape(batch_size, time_steps_q, d_model), the decoded.
            self_attn_weights (Tensor), shape(batch_size, n_heads, time_steps_q, time_steps_q), decoder self attention.
            cross_attn_weights (Tensor), shape(batch_size, n_heads, time_steps_q, time_steps_k), decoder-encoder cross attention.
        """
        tq = q.shape[1]
        no_future_mask = paddle.tril(paddle.ones([tq, tq])) #(tq, tq)
        combined_mask = combine_mask(decoder_mask, no_future_mask)
        
        # pre norm
        q_in = q
        q = self.layer_norm1(q)
        context_vector, self_attn_weights = self.self_mha(q, q, q, combined_mask)
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

    def forward(self, x, mask):
        self_attention_weights = []
        cross_attention_weights = []
        for layer in self:
            x, self_attention_weights_i, cross_attention_weights_i = layer(x, mask)
            self_attention_weights.append(self_attention_weights_i)
            cross_attention_weights.append(cross_attention_weights_i)
        return x, self_attention_weights, cross_attention_weights
    
    
class DecoderPreNet(nn.Layer):
    def __init__(self, d_model, d_hidden, dropout):
        self.lin1 = nn.Linear(d_model, d_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(d_hidden, d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        # the original code said also use dropout in inference
        return self.dropout2(F.relu(self.lin2(self.dropout1(F.relu(self.lin1(x))))))


class PostNet(nn.Layer):
    def __init__(self, d_input, d_hidden, d_output, kernel_size, n_layers):
        self.convs = nn.LayerList()
        kernel_size = kernel_size if isinstance(tuple, kernel_size) else (kernel_size, ) 
        padding = (kernel_size[0] - 1, 0)
        for i in range(n_layers):
            c_in = d_input if i == 0 else d_hidden
            c_out = d_output if i == n_layers - 1 else d_hidden
            self.convs.append(
                Conv1dBatchNorm(c_in, c_out, kernel_size, padding=padding))
        self.last_norm = nn.BatchNorm1d(d_output)
    
    def forward(self, x):
        for layer in self.convs:
            x = paddle.tanh(layer(x))
        x = self.last_norm(x)
        return x


class TransformerTTS(nn.Layer):
    def __init__(self, vocab_size, padding_idx, d_model, d_mel, n_heads, d_ffn, 
                 encoder_layers, decoder_layers, d_prenet, d_postnet, postnet_layers, 
                 postnet_kernel_size, reduction_factor, dropout):
        self.encoder_prenet = nn.Embedding(vocab_size, d_model, padding_idx)
        self.encoder = TransformerEncoder(d_model, n_heads, d_ffn, encoder_layers, dropout)
        self.decoder_prenet = DecoderPreNet(d_model, d_prenet, dropout)
        self.decoder = TransformerDecoder(d_model, n_heads, d_ffn, decoder_layers, dropout)
        self.decoder_postnet = nn.Linear(d_model, reduction_factor * d_mel)
        self.postnet = PostNet(d_mel, d_postnet, d_mel, postnet_kernel_size, postnet_layers)
    
    def forward(self):
        pass
    
    def infer(self):
        pass