import math

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I

from parakeet.modules.attention import _split_heads, _concat_heads


class GST(nn.Layer):
    def __init__(self, n_mels, conv_hidden_sizes, conv_kernel_size,
                 conv_stride, reference_dim, num_style_tokens, num_heads,
                 gst_embedding_size):
        super().__init__()
        self.encoder = ReferenceEncoder(n_mels, conv_hidden_sizes,
                                        conv_kernel_size, conv_stride,
                                        reference_dim)
        self.style_token_layer = StyleTokenLayer(reference_dim,
                                                 num_style_tokens, num_heads,
                                                 gst_embedding_size)

    def forward(self, mel):
        enc_out = self.encoder(mel)
        style_embed = self.style_token_layer(enc_out)
        return style_embed


class ReferenceEncoder(nn.Layer):
    """Reference Encoder in `Style Tokens: Unsupervised Style Modeling, Control 
    and Transfer in End-to-End Speech Synthesis<http://arxiv.org/abs/1803.09017>_"""
    def __init__(self,
                 input_size,
                 hidden_sizes=[32, 32, 64, 64, 128, 128],
                 kernel_size=(3, 3),
                 stride=(2, 2),
                 output_size=128):
        super().__init__()
        in_channels = [1] + hidden_sizes[:-1]
        convs = []
        for c_in, c_out in zip(in_channels, hidden_sizes):
            conv = nn.Conv2D(c_in,
                             c_out,
                             kernel_size,
                             stride=stride,
                             padding="same")
            bn = nn.BatchNorm2D(c_out)
            act = nn.ReLU()
            convs.extend([conv, bn, act])
        self.convs = nn.Sequential(*convs)

        post_conv_width = self._compute_conv_output_size(
            input_size, kernel_size[1], stride[1], (kernel_size[1] - 1) // 2,
            len(hidden_sizes))
        self.rnn = nn.GRU(hidden_sizes[-1] * post_conv_width, output_size)

    def forward(self, mel):
        # mel: (B, T, C)
        x = mel.unsqueeze(1)  # (B, 1, T, C)
        x = self.convs(x)  # (B, hidden_sizes[-1], T', C')

        batch_size = x.shape[0]
        post_conv_time_stpes = x.shape[2]

        x = paddle.transpose(x, [0, 2, 1, 3])
        x = paddle.reshape(x, [batch_size, post_conv_time_stpes, -1])
        memory, h = self.rnn(x)
        # h: [num_layers=1, batch_size, output_size)
        embed = h.squeeze(0)
        return embed

    @staticmethod
    def _compute_conv_output_size(size, kernel_size, stride, padding,
                                  num_layers):
        for _ in range(num_layers):
            size = (size + 2 * padding - kernel_size) // stride + 1
        return size


class StyleTokenLayer(nn.Layer):
    """Style Token Layer in `Style Tokens: Unsupervised Style Modeling, Control 
    and Transfer in End-to-End Speech Synthesis<http://arxiv.org/abs/1803.09017>_"""
    def __init__(self,
                 input_size,
                 num_style_tokens,
                 num_heads,
                 embedding_size=256):
        super().__init__()

        key_dim = embedding_size // num_heads
        self.style_tokens = self.create_parameter([num_style_tokens, key_dim],
                                                  attr=I.Normal(std=0.5))
        self.attention = MultiHeadAttention(input_size, key_dim, num_heads,
                                            embedding_size)

    def forward(self, embed):
        batch_size = embed.shape[0]
        # (B, 1, Cq)
        embed = embed.unsqueeze(1)
        # (B, N, Ck)
        tokens = paddle.tanh(self.style_tokens)\
            .unsqueeze(0)\
            .expand([batch_size, -1, -1])
        style_embed = self.attention(embed, tokens)
        style_embed = style_embed.squeeze(1)
        return style_embed


class MultiHeadAttention(nn.Layer):
    """MultiHead Attention in `Style Tokens: Unsupervised Style Modeling, Control 
    and Transfer in End-to-End Speech Synthesis<http://arxiv.org/abs/1803.09017>_"""
    def __init__(self, query_dim, key_dim, num_heads, hidden_dim):
        super().__init__()
        self.wq = nn.Linear(query_dim, hidden_dim, bias_attr=False)
        self.wk = nn.Linear(key_dim, hidden_dim, bias_attr=False)
        self.wv = nn.Linear(key_dim, hidden_dim, bias_attr=False)

        self.num_heads = num_heads
        self.depth = hidden_dim // num_heads

    def forward(self, query, key):
        # (B, h, T, C)
        queries = _split_heads(self.wq(query), self.num_heads)
        keys = _split_heads(self.wk(key), self.num_heads)
        values = _split_heads(self.wv(key), self.num_heads)

        energy = paddle.matmul(queries, keys, transpose_y=True) / math.sqrt(
            self.depth)
        attention_weights = F.softmax(energy)
        context_vectors = paddle.matmul(attention_weights, values)
        context_vectors = _concat_heads(context_vectors)
        return context_vectors
