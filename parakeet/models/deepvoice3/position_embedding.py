import numpy as np
from paddle import fluid
import paddle.fluid.layers as F
import paddle.fluid.dygraph as dg


def compute_position_embedding(radians, speaker_position_rate):
    """compute sin/cos separately and scatter them to a zero.
    
    Arguments:
        radians {Variable} -- shape(n_vocab, embed_dim), the radians matrix.
        speaker_position_rate {Variable} -- shape(batch_size, ), speaker positioning rate.
    
    Returns:
        Variable -- shape(batch_size, n_vocab, embed_dim), the sin, cos matrix.
    """
    _, embed_dim = radians.shape
    batch_size = speaker_position_rate.shape[0]
    speaker_position_rate = F.unsqueeze(speaker_position_rate, [1, 2])
    scaled_radians = speaker_position_rate * radians

    odd_mask = (np.arange(embed_dim) % 2).astype(np.float32)
    odd_mask = dg.to_variable(odd_mask)

    out = odd_mask * F.cos(scaled_radians) \
        + (1 - odd_mask) * F.sin(scaled_radians)
    out = F.concat(
        [F.zeros((batch_size, 1, embed_dim), radians.dtype), out[:, 1:, :]],
        axis=1)
    return out


def position_encoding_init(n_position,
                           d_pos_vec,
                           position_rate=1.0,
                           padding_idx=None):
    """init the position encoding table"""
    # keep idx 0 for padding token position encoding zero vector
    # CAUTION: it is radians here, sin and cos are not applied
    # CAUTION: difference here
    indices_range = np.expand_dims(np.arange(n_position), -1)
    embed_range = 2 * (np.arange(d_pos_vec) // 2)
    radians = position_rate \
            * indices_range \
            / np.power(1.e4, embed_range / d_pos_vec)
    if padding_idx is not None:
        radians[padding_idx] = 0.
    return radians


class PositionEmbedding(dg.Layer):
    def __init__(self,
                 n_position,
                 d_pos_vec,
                 position_rate=1.0,
                 param_attr=None,
                 max_norm=None,
                 padding_idx=None):
        super(PositionEmbedding, self).__init__()
        self.weight = self.create_parameter((n_position, d_pos_vec))
        self.weight.set_value(
            position_encoding_init(n_position, d_pos_vec, position_rate,
                                   padding_idx).astype("float32"))

    def forward(self, indices, speaker_position_rate=None):
        """
        Args:
            indices (Variable): Shape (B, T), dtype: int64, position
                indices, where B means the batch size, T means the time steps.
            speaker_position_rate (Variable | float, optional), position
                rate. It can be a float point number or a Variable with 
                shape (1,), then this speaker_position_rate is used for every 
                example. It can also be a Variable with shape (B, 1), which 
                contains a speaker position rate for each speaker.
        Returns:
            out (Variable): Shape(B, T, C_pos), position embedding, where C_pos 
                means position embedding size.
        """
        batch_size, time_steps = indices.shape

        # convert speaker_position_rate to a Variable with shape(B, )
        if isinstance(speaker_position_rate, float):
            speaker_position_rate = dg.to_variable(
                np.array([speaker_position_rate]).astype("float32"))
            speaker_position_rate = F.expand(speaker_position_rate,
                                             [batch_size])
        elif isinstance(speaker_position_rate, fluid.framework.Variable) \
            and list(speaker_position_rate.shape) == [1]:
            speaker_position_rate = F.expand(speaker_position_rate,
                                             [batch_size])
        assert len(speaker_position_rate.shape) == 1 and \
            list(speaker_position_rate.shape) == [batch_size]

        weight = compute_position_embedding(self.weight,
                                            speaker_position_rate)  # (B, V, C)
        # make indices for gather_nd
        batch_id = F.expand(
            F.unsqueeze(F.range(0, batch_size, 1, dtype="int64"), [1]),
            [1, time_steps])
        # (B, T, 2)
        gather_nd_id = F.stack([batch_id, indices], -1)

        out = F.gather_nd(weight, gather_nd_id)
        return out