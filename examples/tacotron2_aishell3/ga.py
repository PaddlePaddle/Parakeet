import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.fluid.layers import sequence_mask


def attention_guide(dec_lens, enc_lens, N, T, g, dtype=None):
    """Build that W matrix. shape(B, T_dec, T_enc)
    W[i, n, t] = 1 - exp(-(n/dec_lens[i] - t/enc_lens[i])**2 / (2g**2))

    See also:
    Tachibana, Hideyuki, Katsuya Uenoyama, and Shunsuke Aihara. 2017. “Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention.” ArXiv:1710.08969 [Cs, Eess], October. http://arxiv.org/abs/1710.08969.

    """
    dtype = dtype or paddle.get_default_dtype()
    dec_pos = paddle.arange(0, N).astype(dtype) / dec_lens.unsqueeze(-1)  # n/N # shape(B, T_dec)
    enc_pos = paddle.arange(0, T).astype(dtype) / enc_lens.unsqueeze(-1)  # t/T # shape(B, T_enc)
    W = 1 - paddle.exp(-(dec_pos.unsqueeze(-1) - enc_pos.unsqueeze(1))**2 / (2 * g ** 2))

    dec_mask = sequence_mask(dec_lens, maxlen=N)
    enc_mask = sequence_mask(enc_lens, maxlen=T)
    mask = dec_mask.unsqueeze(-1) * enc_mask.unsqueeze(1)
    mask = paddle.cast(mask, W.dtype)

    W *= mask
    return W


def guided_attention_loss(attention_weight, dec_lens, enc_lens, g):
    """Guided attention loss, masked to excluded padding parts."""
    _, N, T = attention_weight.shape
    W = attention_guide(dec_lens, enc_lens, N, T, g, attention_weight.dtype)

    total_tokens = (dec_lens * enc_lens).astype(W.dtype)
    loss = paddle.mean(paddle.sum(W * attention_weight, [1, 2]) / total_tokens)
    return loss, W
