# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
from parakeet.models.transformer_tts.utils import *
from parakeet.modules.multihead_attention import MultiheadAttention
from parakeet.modules.ffn import PositionwiseFeedForward
from parakeet.models.transformer_tts.encoderprenet import EncoderPrenet


class Encoder(dg.Layer):
    def __init__(self, embedding_size, num_hidden, num_head=4, n_layers=3):
        """Encoder layer of TransformerTTS.

        Args:
            embedding_size (int): the size of position embedding.
            num_hidden (int): the size of hidden layer in network.
            num_head (int, optional): the head number of multihead attention. Defaults to 4.
            n_layers (int, optional): the layers number of multihead attention. Defaults to 3.
        """
        super(Encoder, self).__init__()
        self.num_hidden = num_hidden
        self.num_head = num_head
        param = fluid.ParamAttr(initializer=fluid.initializer.Constant(
            value=1.0))
        self.alpha = self.create_parameter(
            shape=(1, ), attr=param, dtype='float32')
        self.pos_inp = get_sinusoid_encoding_table(
            1024, self.num_hidden, padding_idx=0)
        self.pos_emb = dg.Embedding(
            size=[1024, num_hidden],
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.NumpyArrayInitializer(
                    self.pos_inp),
                trainable=False))
        self.encoder_prenet = EncoderPrenet(
            embedding_size=embedding_size,
            num_hidden=num_hidden,
            use_cudnn=True)
        self.layers = [
            MultiheadAttention(num_hidden, num_hidden // num_head,
                               num_hidden // num_head) for _ in range(n_layers)
        ]
        for i, layer in enumerate(self.layers):
            self.add_sublayer("self_attn_{}".format(i), layer)
        self.ffns = [
            PositionwiseFeedForward(
                num_hidden,
                num_hidden * num_head,
                filter_size=1,
                use_cudnn=True) for _ in range(n_layers)
        ]
        for i, layer in enumerate(self.ffns):
            self.add_sublayer("ffns_{}".format(i), layer)

    def forward(self, x, positional):
        """
        Encode text sequence.
        
        Args:
            x (Variable): shape(B, T_text), dtype float32, the input character,
                where T_text means the timesteps of input text,
            positional (Variable): shape(B, T_text), dtype int64, the characters position. 
                
        Returns:
            x (Variable): shape(B, T_text, C), the encoder output.
            attentions (list[Variable]): len(n_layers), the encoder self attention list.
        """

        # Encoder pre_network
        x = self.encoder_prenet(x)

        if fluid.framework._dygraph_tracer()._train_mode:
            mask = get_attn_key_pad_mask(positional, self.num_head, x.dtype)
            query_mask = get_non_pad_mask(positional, self.num_head, x.dtype)

        else:
            query_mask, mask = None, None

        # Get positional encoding
        positional = self.pos_emb(positional)

        x = positional * self.alpha + x

        # Positional dropout
        x = layers.dropout(x, 0.1, dropout_implementation='upscale_in_train')

        # Self attention encoder
        attentions = list()
        for layer, ffn in zip(self.layers, self.ffns):
            x, attention = layer(x, x, x, mask=mask, query_mask=query_mask)
            x = ffn(x)
            attentions.append(attention)

        return x, attentions, query_mask
