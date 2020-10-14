import math
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as I

from parakeet.modules.conv import Conv1dBatchNorm


class Highway(nn.Layer):
    def __init__(self, num_features):
        super(Highway, self).__init__()
        self.H = nn.Linear(num_features, num_features)
        self.T = nn.Linear(num_features, num_features,
                           bias_attr=I.Constant(-1.))

        self.num_features = num_features

    def forward(self, x):
        H = F.relu(self.H(x))
        T = F.sigmoid(self.T(x))  # gate
        return H * T + x * (1.0 - T)


class CBHG(nn.Layer):
    def __init__(self, in_channels, out_channels_per_conv, max_kernel_size,
                 projection_channels,
                 num_highways, highway_features,
                 gru_features):
        super(CBHG, self).__init__()
        self.conv1d_banks = nn.LayerList(
            [Conv1dBatchNorm(in_channels, out_channels_per_conv, (k,),
                             padding=((k - 1) // 2, k // 2))
             for k in range(1, 1 + max_kernel_size)])

        self.projections = nn.LayerList()
        projection_channels = list(projection_channels)
        proj_in_channels = [max_kernel_size *
                            out_channels_per_conv] + projection_channels
        proj_out_channels = projection_channels + \
            [in_channels]  # ensure residual connection
        for c_in, c_out in zip(proj_in_channels, proj_out_channels):
            conv = nn.Conv1d(c_in, c_out, (3,), padding=(1, 1))
            self.projections.append(conv)

        if in_channels != highway_features:
            self.pre_highway = nn.Linear(in_channels, highway_features)

        self.highways = nn.LayerList(
            [Highway(highway_features) for _ in range(num_highways)])

        self.gru = nn.GRU(highway_features, gru_features,
                          direction="bidirectional")

        self.in_channels = in_channels
        self.out_channels_per_conv = out_channels_per_conv
        self.max_kernel_size = max_kernel_size
        self.num_projections = 1 + len(projection_channels)
        self.num_highways = num_highways
        self.highway_features = highway_features
        self.gru_features = gru_features

    def forward(self, x):
        input = x

        # conv banks
        conv_outputs = []
        for conv in self.conv1d_banks:
            conv_outputs.append(conv(x))
        x = F.relu(paddle.concat(conv_outputs, 1))

        # max pool
        x = F.max_pool1d(x, 2, stride=1, padding=(0, 1))

        # conv1d projections
        n_projections = len(self.projections)
        for i, conv in enumerate(self.projections):
            x = conv(x)
            if i != n_projections:
                x = F.relu(x)
        x += input  # residual connection

        # highway
        x = paddle.transpose(x, [0, 2, 1])
        if hasattr(self, "pre_highway"):
            x = self.pre_highway(x)

        # gru
        x, _ = self.gru(x)
        return x
