import numpy as np

from paddle import fluid
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as F
import paddle.fluid.initializer as I

from parakeet.modules.weight_norm import Conv1D, Conv1DCell, Conv2D, Linear


class Conv1DGLU(dg.Layer):
    """
    A Convolution 1D block with GLU activation. It also applys dropout for the 
    input x. It fuses speaker embeddings through a FC activated by softsign. It
    has residual connection from the input x, and scale the output by 
    np.sqrt(0.5).
    """
    def __init__(self,
                 n_speakers,
                 speaker_dim,
                 in_channels,
                 num_filters,
                 filter_size=1,
                 dilation=1,
                 std_mul=4.0,
                 dropout=0.0,
                 causal=False,
                 residual=True):
        super(Conv1DGLU, self).__init__()

        # conv spec
        self.in_channels = in_channels
        self.n_speakers = n_speakers
        self.speaker_dim = speaker_dim
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.dilation = dilation

        # padding
        self.causal = causal

        # weight init and dropout
        self.std_mul = std_mul
        self.dropout = dropout
        c_in = filter_size * in_channels
        std = np.sqrt(std_mul * (1 - dropout) / c_in)

        self.residual = residual
        if residual:
            assert (
                in_channels == num_filters
            ), "this block uses residual connection"\
                "the input_channes should equals num_filters"
        self.conv = Conv1DCell(in_channels,
                               2 * num_filters,
                               filter_size,
                               dilation,
                               causal,
                               param_attr=I.Normal(scale=std))

        if n_speakers > 1:
            assert (speaker_dim is not None
                    ), "speaker embed should not be null in multi-speaker case"
            std = np.sqrt(1 / speaker_dim)
            self.fc = Linear(speaker_dim,
                             num_filters,
                             param_attr=I.Normal(scale=std))

    def forward(self, x, speaker_embed=None):
        """
        Args:
            x (Variable): Shape(B, C_in, T), the input of Conv1DGLU
                layer, where B means batch_size, C_in means the input channels
                T means input time steps.
            speaker_embed_bct1 (Variable): Shape(B, C_sp), expanded
                speaker embed, where C_sp means speaker embedding size. Note
                that when using residual connection, the Conv1DGLU does not
                change the number of channels, so out channels equals input
                channels.

        Returns:
            x (Variable): Shape(B, C_out, T), the output of Conv1DGLU, where
                C_out means the output channels of Conv1DGLU.
        """
        residual = x
        x = F.dropout(x,
                      self.dropout,
                      dropout_implementation="upscale_in_train")
        x = self.conv(x)
        content, gate = F.split(x, num_or_sections=2, dim=1)

        if speaker_embed is not None:
            sp = F.softsign(self.fc(speaker_embed))
            content = F.elementwise_add(content, sp, axis=0)

        # glu
        x = F.sigmoid(gate) * content

        if self.residual:
            x = F.scale(x + residual, np.sqrt(0.5))
        return x

    def start_sequence(self):
        self.conv.start_sequence()

    def add_input(self, x_t, speaker_embed=None):
        """
        Args:
            x (Variable): Shape(B, C_in), the input of Conv1DGLU
                layer, where B means batch_size, C_in means the input channels.
            speaker_embed_bct1 (Variable): Shape(B, C_sp), expanded
                speaker embed, where C_sp means speaker embedding size. Note
                that when using residual connection, the Conv1DGLU does not
                change the number of channels, so out channels equals input
                channels.

        Returns:
            x (Variable): Shape(B, C_out), the output of Conv1DGLU, where
                C_out means the output channels of Conv1DGLU.
        """
        residual = x_t
        x_t = F.dropout(x_t,
                        self.dropout,
                        dropout_implementation="upscale_in_train")
        x_t = self.conv.add_input(x_t)
        content_t, gate_t = F.split(x_t, num_or_sections=2, dim=1)

        if speaker_embed is not None:
            sp = F.softsign(self.fc(speaker_embed))
            content_t = F.elementwise_add(content_t, sp, axis=0)

        # glu
        x_t = F.sigmoid(gate_t) * content_t

        if self.residual:
            x_t = F.scale(x_t + residual, np.sqrt(0.5))
        return x_t
