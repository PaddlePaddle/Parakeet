import math
import numpy as np
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers

class ScaledDotProductAttention(dg.Layer):
    def __init__(self, d_key):
        super(ScaledDotProductAttention, self).__init__()

        self.d_key = d_key
    
    # please attention this mask is diff from pytorch
    def forward(self, key, value, query, mask=None, query_mask=None):
        # Compute attention score
        attention = layers.matmul(query, key, transpose_y=True) #transpose the last dim in y
        attention = attention / math.sqrt(self.d_key)

        # Mask key to ignore padding
        if mask is not None:
            attention = attention * (mask == 0).astype(np.float32)
            mask = mask * (-2 ** 32 + 1)
            attention = attention + mask
            

        attention = layers.softmax(attention)
        attention = layers.dropout(attention, 0.0)
        # Mask query to ignore padding
        # Not sure how to work
        if query_mask is not None:
            attention = attention * query_mask
        
        result = layers.matmul(attention, value)
        return result, attention

class MultiheadAttention(dg.Layer):
    def __init__(self, num_hidden, d_k, d_q, num_head=4, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.num_hidden = num_hidden
        self.num_head = num_head
        self.d_k = d_k
        self.d_q = d_q
        self.dropout = dropout

        self.key = dg.Linear(num_hidden, num_head * d_k)
        self.value = dg.Linear(num_hidden, num_head * d_k)
        self.query = dg.Linear(num_hidden, num_head * d_q)

        self.scal_attn = ScaledDotProductAttention(d_k)

        self.fc = dg.Linear(num_head * d_q, num_hidden)

        self.layer_norm = dg.LayerNorm(num_hidden)

    def forward(self, key, value, query_input, mask=None, query_mask=None):
        batch_size = key.shape[0]
        seq_len_key = key.shape[1]
        seq_len_query = query_input.shape[1]

        # repeat masks h times
        if query_mask is not None:
            query_mask = layers.expand(query_mask, [self.num_head, 1, seq_len_key])
        if mask is not None:
            mask = layers.expand(mask, (self.num_head, 1, 1))
        
        # Make multihead attention
        # key & value.shape = (batch_size, seq_len, feature)(feature = num_head * num_hidden_per_attn)
        key = layers.reshape(self.key(key), [batch_size, seq_len_key, self.num_head, self.d_k])
        value = layers.reshape(self.value(value), [batch_size, seq_len_key, self.num_head, self.d_k])
        query = layers.reshape(self.query(query_input), [batch_size, seq_len_query, self.num_head, self.d_q])

        key = layers.reshape(layers.transpose(key, [2, 0, 1, 3]), [-1, seq_len_key, self.d_k])
        value = layers.reshape(layers.transpose(value, [2, 0, 1, 3]), [-1, seq_len_key, self.d_k])
        query = layers.reshape(layers.transpose(query, [2, 0, 1, 3]), [-1, seq_len_query, self.d_q])
        result, attention = self.scal_attn(key, value, query, mask=mask, query_mask=query_mask)
        
        # concat all multihead result
        result = layers.reshape(result, [self.num_head, batch_size, seq_len_query, self.d_q])
        result = layers.reshape(layers.transpose(result, [1,2,0,3]),[batch_size, seq_len_query, -1])
        
        result = layers.dropout(self.fc(result), self.dropout)
        result = result + query_input
        
        result = self.layer_norm(result)
        return result, attention