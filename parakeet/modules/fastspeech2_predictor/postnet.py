# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import six
from paddle import nn


class Postnet(nn.Layer):
    """Postnet module for Spectrogram prediction network.

    This is a module of Postnet in Spectrogram prediction network,
    which described in `Natural TTS Synthesis by
    Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The Postnet predicts refines the predicted
    Mel-filterbank of the decoder,
    which helps to compensate the detail sturcture of spectrogram.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(
            self,
            idim,
            odim,
            n_layers=5,
            n_chans=512,
            n_filts=5,
            dropout_rate=0.5,
            use_batch_norm=True, ):
        """Initialize postnet module.

        Parameters
        ----------
        idim : int
            Dimension of the inputs.
        odim : int
            Dimension of the outputs.
        n_layers : int, optional
            The number of layers.
        n_filts : int, optional
            The number of filter size.
        n_units : int, optional
            The number of filter channels.
        use_batch_norm : bool, optional
            Whether to use batch normalization..
        dropout_rate : float, optional
            Dropout rate..
        """
        super(Postnet, self).__init__()
        self.postnet = nn.LayerList()
        for layer in six.moves.range(n_layers - 1):
            ichans = odim if layer == 0 else n_chans
            ochans = odim if layer == n_layers - 1 else n_chans
            if use_batch_norm:
                self.postnet.append(
                    nn.Sequential(
                        nn.Conv1D(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias_attr=False, ),
                        nn.BatchNorm1D(ochans),
                        nn.Tanh(),
                        nn.Dropout(dropout_rate), ))
            else:
                self.postnet.append(
                    nn.Sequential(
                        nn.Conv1D(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias_attr=False, ),
                        nn.Tanh(),
                        nn.Dropout(dropout_rate), ))
        ichans = n_chans if n_layers != 1 else odim
        if use_batch_norm:
            self.postnet.append(
                nn.Sequential(
                    nn.Conv1D(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias_attr=False, ),
                    nn.BatchNorm1D(odim),
                    nn.Dropout(dropout_rate), ))
        else:
            self.postnet.append(
                nn.Sequential(
                    nn.Conv1D(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias_attr=False, ),
                    nn.Dropout(dropout_rate), ))

    def forward(self, xs):
        """Calculate forward propagation.

        Parameters
        ----------
        xs : Tensor
            Batch of the sequences of padded input tensors (B, idim, Tmax).

        Returns
        ----------
        Tensor
            Batch of padded output tensor. (B, odim, Tmax).

        """
        for i in six.moves.range(len(self.postnet)):
            xs = self.postnet[i](xs)
        return xs
