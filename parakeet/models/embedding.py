from paddle import nn
from typing import Optional


class TextEmbedding(nn.Layer):
    """A embedding layer for TTS encoder. It has text embedding and an optioanl
    tone embedding. 
    
    The two embeddings have same feature sizes and are added by defualt. When 
    the two embeddings have different feature sizes, they are concatenated.
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_size: int,
                 tone_vocab_size: Optional[int] = None,
                 tone_embedding_size: Optional[int] = None,
                 padding_idx: Optional[int] = None,
                 tone_padding_idx: Optional[int] = None,
                 concat: bool = False):
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size,
                                           embedding_size,
                                           padding_idx,
                                           weight_attr=I.Normal(0, 1))
        if tone_vocab_size:
            tone_embedding_size = tone_embedding_size or embedding_size
            if tone_embedding_size != embedding_size and not concat:
                raise ValueError(
                    "embedding size != tone_embedding size, only conat is avaiable."
                )
            self.tone_embedding = nn.Embedding(tone_vocab_size,
                                               tone_embedding_size,
                                               tone_padding_idx)
        self.concat = concat

    def forward(self, text, tone=None):
        # shape_hint: [B, T] -> [B, T, C]
        text_embed = self.text_embedding(text)
        if tone is None:
            return text_embed
        
        tone_embed = self.tone_embedding(tone)
        embed = paddle.concat([text_embed, tone_embed], -1) if self.concat \
            else text_embed + tone_embed
        return embed
