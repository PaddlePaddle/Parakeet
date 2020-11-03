import math
import paddle
from paddle import nn
from paddle.nn import functional as F

from parakeet.modules import attention as attn
from parakeet.modules.masking import combine_mask
class PositionwiseFFN(nn.Layer):
    """
    A faithful implementation of Position-wise Feed-Forward Network 
    in `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_.
    It is basically a 3-layer MLP, with relu actication and dropout in between.
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 dropout=0.0):
        """
        Args:
            input_size (int): the input feature size.
            hidden_size (int): the hidden layer's feature size.
            dropout (float, optional): probability of dropout applied to the 
                output of the first fully connected layer. Defaults to 0.0.
        """
        super(PositionwiseFFN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)

        self.input_size = input_size
        self.hidden_szie = hidden_size

    def forward(self, x):
        """positionwise feed forward network.

        Args:
            x (Tensor): shape(*, input_size), the input tensor.

        Returns:
            Tensor: shape(*, input_size), the output tensor.
        """
        l1 = self.dropout(F.relu(self.linear1(x)))
        l2 = self.linear2(l1)
        return l2


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
        self.self_mha = attn.MultiheadAttention(d_model, n_heads, dropout)
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
        context_vector, attn_weights = self.self_mha(x, x, x, paddle.unsqueeze(mask, 1))
        x = self.layer_norm1(x + context_vector)
        
        x = self.layer_norm2(x + self.ffn(x))
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
        self.self_mha = attn.MultiheadAttention(d_model, n_heads, dropout)
        self.layer_norm1 = nn.LayerNorm([d_model], epsilon=1e-6)
        
        self.cross_mha = attn.MultiheadAttention(d_model, n_heads, dropout)
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
        combined_mask = combine_mask(decoder_mask.unsqueeze(1), no_future_mask)
        context_vector, self_attn_weights = self.self_mha(q, q, q, combined_mask)
        q = self.layer_norm1(q + context_vector)
        
        context_vector, cross_attn_weights = self.cross_mha(q, k, v, paddle.unsqueeze(encoder_mask, 1))
        q = self.layer_norm2(q + context_vector)
        
        q = self.layer_norm3(q + self.ffn(q))
        return q, self_attn_weights, cross_attn_weights
