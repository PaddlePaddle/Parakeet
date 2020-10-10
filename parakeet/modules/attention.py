import math
import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F

def scaled_dot_product_attention(q, k, v, mask=None, dropout=0.0, training=True):
    """
    scaled dot product attention with mask. Assume q, k, v all have the same 
    leader dimensions(denoted as * in descriptions below). Dropout is applied to 
    attention weights before weighted sum of values.

    Args:
        q (Tensor): shape(*, T_q, d), the query tensor.
        k (Tensor): shape(*, T_k, d), the key tensor.
        v (Tensor): shape(*, T_k, d_v), the value tensor.
        mask (Tensor, optional): shape(*, T_q, T_k) or broadcastable shape, the 
            mask tensor, 0 correspond to padding. Defaults to None.
    
    Returns:
        (out, attn_weights)
        out (Tensor): shape(*, T_q, d_v), the context vector.
        attn_weights (Tensor): shape(*, T_q, T_k), the attention weights.
    """
    d = q.shape[-1] # we only support imperative execution
    qk = paddle.matmul(q, k, transpose_y=True)
    scaled_logit = paddle.scale(qk, 1.0 / math.sqrt(d))
    
    if mask is not None:
        scaled_logit += paddle.scale((1.0 - mask), -1e12) # hard coded here
    
    attn_weights = F.softmax(scaled_logit, axis=-1)
    attn_weights = F.dropout(attn_weights, dropout, training=training)
    out = paddle.matmul(attn_weights, v)
    return out, attn_weights

def drop_head(x, drop_n_heads, training):
    """
    Drop n heads from multiple context vectors.

    Args:
        x (Tensor): shape(batch_size, num_heads, time_steps, channels), the input.
        drop_n_heads (int): [description]
        training ([type]): [description]

    Returns:
        [type]: [description]
    """
    if not training or (drop_n_heads == 0):
        return x
    
    batch_size, num_heads, _, _ = x.shape
    # drop all heads
    if num_heads == drop_n_heads:
        return paddle.zeros_like(x)
    
    mask = np.ones([batch_size, num_heads])
    mask[:, :drop_n_heads] = 0
    for subarray in mask:
        np.random.shuffle(subarray)
    scale = float(num_heads) / (num_heads - drop_n_heads)
    mask = scale * np.reshape(mask, [batch_size, num_heads, 1, 1])
    out = x * paddle.to_tensor(mask)
    return out

def _split_heads(x, num_heads):
    batch_size, time_steps, _ = x.shape
    x = paddle.reshape(x, [batch_size, time_steps, num_heads, -1])
    x = paddle.transpose(x, [0, 2, 1, 3])
    return x

def _concat_heads(x):
    batch_size, _, time_steps, _ = x.shape
    x = paddle.transpose(x, [0, 2, 1, 3])
    x = paddle.reshape(x, [batch_size, time_steps, -1])
    return x

# Standard implementations of Monohead Attention & Multihead Attention
class MonoheadAttention(nn.Layer):
    def __init__(self, model_dim, dropout=0.0, k_dim=None, v_dim=None):
        """
        Monohead Attention module.

        Args:
            model_dim (int): the feature size of query.
            dropout (float, optional): dropout probability of scaled dot product
                attention and final context vector. Defaults to 0.0.
            k_dim (int, optional): feature size of the key of each scaled dot 
                product attention. If not provided, it is set to 
                model_dim / num_heads. Defaults to None.
            v_dim (int, optional): feature size of the key of each scaled dot 
                product attention. If not provided, it is set to 
                model_dim / num_heads. Defaults to None.
        """
        super(MonoheadAttention, self).__init__()
        k_dim = k_dim or model_dim
        v_dim = v_dim or model_dim
        self.affine_q = nn.Linear(model_dim, k_dim)
        self.affine_k = nn.Linear(model_dim, k_dim)
        self.affine_v = nn.Linear(model_dim, v_dim)
        self.affine_o = nn.Linear(v_dim, model_dim)
        
        self.model_dim = model_dim
        self.dropout = dropout
    
    def forward(self, q, k, v, mask):
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
        q = self.affine_q(q) # (B, T, C)
        k = self.affine_k(k)
        v = self.affine_v(v)
        
        context_vectors, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, self.dropout, self.training)
        
        out = self.affine_o(context_vectors)
        return out, attention_weights

        
class MultiheadAttention(nn.Layer):
    """
    Multihead scaled dot product attention.
    """
    def __init__(self, model_dim, num_heads, dropout=0.0, k_dim=None, v_dim=None):
        """
        Multihead Attention module.

        Args:
            model_dim (int): the feature size of query.
            num_heads (int): the number of attention heads.
            dropout (float, optional): dropout probability of scaled dot product
                attention and final context vector. Defaults to 0.0.
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
        self.affine_o = nn.Linear(num_heads * v_dim, model_dim)
        
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.dropout = dropout
    
    def forward(self, q, k, v, mask):
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
        q = _split_heads(self.affine_q(q), self.num_heads) # (B, h, T, C)
        k = _split_heads(self.affine_k(k), self.num_heads)
        v = _split_heads(self.affine_v(v), self.num_heads)
        mask = paddle.unsqueeze(mask, 1) # unsqueeze for the h dim
        
        context_vectors, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, self.dropout, self.training)
        # NOTE: there is more sophisticated implementation: Scheduled DropHead
        context_vectors = _concat_heads(context_vectors) # (B, T, h*C)
        out = self.affine_o(context_vectors)
        return out, attention_weights
