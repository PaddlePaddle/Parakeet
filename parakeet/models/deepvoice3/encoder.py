import numpy as np
from collections import namedtuple

import paddle.fluid.layers as F
import paddle.fluid.initializer as I
import paddle.fluid.dygraph as dg

from parakeet.modules.weight_norm import Conv1D, Linear
from parakeet.models.deepvoice3.conv1dglu import Conv1DGLU

ConvSpec = namedtuple("ConvSpec", ["out_channels", "filter_size", "dilation"])


class Encoder(dg.Layer):
    def __init__(self,
                 n_vocab,
                 embed_dim,
                 n_speakers,
                 speaker_dim,
                 padding_idx=None,
                 embedding_weight_std=0.1,
                 convolutions=(ConvSpec(64, 5, 1), ) * 7,
                 max_positions=512,
                 dropout=0.):
        super(Encoder, self).__init__()

        self.embedding_weight_std = embedding_weight_std
        self.embed = dg.Embedding(
            (n_vocab, embed_dim),
            padding_idx=padding_idx,
            param_attr=I.Normal(scale=embedding_weight_std))

        self.dropout = dropout
        if n_speakers > 1:
            std = np.sqrt((1 - dropout) / speaker_dim)
            self.sp_proj1 = Linear(speaker_dim,
                                   embed_dim,
                                   act="softsign",
                                   param_attr=I.Normal(scale=std))
            self.sp_proj2 = Linear(speaker_dim,
                                   embed_dim,
                                   act="softsign",
                                   param_attr=I.Normal(scale=std))
        self.n_speakers = n_speakers

        self.convolutions = dg.LayerList()
        in_channels = embed_dim
        std_mul = 1.0
        for (out_channels, filter_size, dilation) in convolutions:
            # 1 * 1 convolution & relu
            if in_channels != out_channels:
                std = np.sqrt(std_mul / in_channels)
                self.convolutions.append(
                    Conv1D(in_channels,
                           out_channels,
                           1,
                           act="relu",
                           param_attr=I.Normal(scale=std)))
                in_channels = out_channels
                std_mul = 2.0

            self.convolutions.append(
                Conv1DGLU(n_speakers,
                          speaker_dim,
                          in_channels,
                          out_channels,
                          filter_size,
                          dilation,
                          std_mul,
                          dropout,
                          causal=False,
                          residual=True))
            in_channels = out_channels
            std_mul = 4.0

        std = np.sqrt(std_mul * (1 - dropout) / in_channels)
        self.convolutions.append(
            Conv1D(in_channels, embed_dim, 1, param_attr=I.Normal(scale=std)))

    def forward(self, x, speaker_embed=None):
        """
        Encode text sequence.
        
        Args:
            x (Variable): Shape(B, T_enc), dtype: int64. Ihe input text
                indices. T_enc means the timesteps of decoder input x.
            speaker_embed (Variable, optional): Shape(batch_size, speaker_dim),
                dtype: float32. Speaker embeddings. This arg is not None only
                when the model is a multispeaker model.

        Returns:
            keys (Variable), Shape(B, T_enc, C_emb), the encoded
                representation for keys, where C_emb menas the text embedding
                size.
            values (Variable), Shape(B, T_enc, C_emb), the encoded
                representation for values.
        """
        x = self.embed(x)
        x = F.dropout(x,
                      self.dropout,
                      dropout_implementation="upscale_in_train")
        x = F.transpose(x, [0, 2, 1])

        if self.n_speakers > 1 and speaker_embed is not None:
            speaker_embed = F.dropout(
                speaker_embed,
                self.dropout,
                dropout_implementation="upscale_in_train")
            x = F.elementwise_add(x, self.sp_proj1(speaker_embed), axis=0)

        input_embed = x
        for layer in self.convolutions:
            if isinstance(layer, Conv1DGLU):
                x = layer(x, speaker_embed)
            else:
                # layer is a Conv1D with (1,) filter wrapped by WeightNormWrapper
                x = layer(x)

        if self.n_speakers > 1 and speaker_embed is not None:
            x = F.elementwise_add(x, self.sp_proj2(speaker_embed), axis=0)

        keys = x  # (B, C, T)
        values = F.scale(input_embed + x, scale=np.sqrt(0.5))
        keys = F.transpose(keys, [0, 2, 1])
        values = F.transpose(values, [0, 2, 1])
        return keys, values
