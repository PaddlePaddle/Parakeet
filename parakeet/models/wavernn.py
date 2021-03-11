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

import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.nn import initializer as IS

from parakeet.utils import checkpoint
from parakeet.modules import geometry as geo
from parakeet.utils.distribution import sample_from_discretized_mix_logistic

from utils.audio import decode_mu_law
from pathlib import Path
import time
import sys


__all__ = ["WaveRNN"]


class ResBlock(nn.Layer):
    def __init__(self, dims):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1D(dims, dims, kernel_size=1, bias_attr=False)
        self.conv2 = nn.Conv1D(dims, dims, kernel_size=1, bias_attr=False)
        self.batch_norm1 = nn.BatchNorm1D(dims)
        self.batch_norm2 = nn.BatchNorm1D(dims)

    def forward(self, x):
        '''
        conv -> bn -> relu -> conv -> bn + residual connection
        '''
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Layer):
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims, pad):
        super(MelResNet, self).__init__()
        k_size = pad * 2 + 1
        # pay attention here, the dim reduces pad * 2
        self.conv_in = nn.Conv1D(in_dims, compute_dims, kernel_size=k_size, bias_attr=False)
        self.batch_norm = nn.BatchNorm1D(compute_dims)
        self.layers = nn.LayerList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1D(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        '''
        conv_in -> bn -> relu -> resblock * n -> conv_out
        :param x: [batch_size, in_dims/80, n_framse]
        :return: x_out: [batch_size, res_out_dims/128, n_frames]
        '''
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(nn.Layer):
    def __init__(self, x_scale, y_scale):
        '''

        :param x_scale: 275
        :param y_scale: 1
        '''
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        '''

        :param x: mel, [batch, 1, 128, seq_len]
        :return: output: [batch, 1, 128, seq_len * x_scale/275]
        '''
        b, c, h, w = x.shape
        # mel: [batch, 1, 128, seq_len] -> [batch, 1, 128, 1, seq_len, 1]
        # x: [b, c, h, w] -> [b, c, h, w, 1] -> [b, c, h, 1, w, 1]
        x = x.unsqueeze(-1).unsqueeze(3)
        # mel: [batch, 1, 128, 1, seq_len, 1] -> [batch, 1, 128, 1, seq_len, 275]
        # x: [b, c, h, 1, w, 1] -> [b, c, h, y_scale, w, x_scale]
        # TODO pay attention function paddle.expand
        x = x.expand([b, c, h, self.y_scale, w, self.x_scale])
        # mel: [batch, 1, 128, 1, seq_len, 275] -> [batch, 1, 128, seq_len * 275]
        # x: [b, c, h, y_scale, w, x_scale] -> [b, c, h * y_scale, w * x_scale]
        return x.reshape([b, c, h * self.y_scale, w * self.x_scale])


class UpsampleNetwork(nn.Layer):
    def __init__(self, feat_dims, upsample_scales, compute_dims, res_blocks, res_out_dims, pad):
        '''
        Args:
            :param feat_dims: num_mels, 80
            :param upsample_scales: [5, 5, 11]
            :param compute_dims: 128
            :param res_blocks: 10
            :param res_out_dims: 128
            :param pad: 2
        '''
        super(UpsampleNetwork, self).__init__()
        # total_scale is the total Up sampling multiple, here is 275
        total_scale = np.cumproduct(upsample_scales)[-1]
        # TODO pad*total_scale is numpy.int64
        self.indent = int(pad * total_scale)
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.LayerList()
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)

            conv = nn.Conv2D(1, 1, kernel_size=k_size, padding=padding, bias_attr=False)
            weight_ = paddle.full_like(conv.weight, 1. / k_size[1])
            conv.weight.set_value(weight_)
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        '''
        :param m: mel, [batch, 80, n_frames]
        :return: mel, aux
        '''
        # aux: [batch_size, 80, n_frames] -> [batch_size, 128, n_frames-2*pad]
        # -> [batch_size, 1, 128, n_frames-2*pad]
        aux = self.resnet(m).unsqueeze(1)
        # aux: [batch_size, 1, 128, n_frames-2*pad] -> [batch_size, 1, 128, n_frames*275-2*pad*175]
        aux = self.resnet_stretch(aux)
        # aux: [batch_size, 1, 128, n_frames*275] -> [batch_size, 128, n_frames*275]
        aux = aux.squeeze(1)
        # m: [batch_size, 80, n_frames] -> [batch_size, 1, 80, n_frames]
        m = m.unsqueeze(1)
        for f in self.up_layers:
            m = f(m)
        # m: [batch_size, 1, 80, n_frames*275] -> [batch_size, 80, n_frames*275]
        # -> [batch_size, 80, n_frames*275-2*pad*275]
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        # m: [batch_size, n_frames*275-2*pad*275, 80]
        # aux: [batch_size, n_frames*275-2*pad*275, 128]
        return m.transpose([0, 2, 1]), aux.transpose([0, 2, 1])


class WaveRNN(nn.Layer):
    def __init__(self, rnn_dims, fc_dims, bits, pad, upsample_factors,
                 feat_dims, compute_dims, res_out_dims, res_blocks,
                 hop_length, sample_rate, mode='RAW'):
        '''
        Args:
            :param rnn_dims: 512
            :param fc_dims: 512
            :param bits: 9
            :param pad: voc_pad 2
            :param upsample_factors: [5, 5, 11]
            :param feat_dims: num_mels = 80
            :param compute_dims: 128
            :param res_out_dims: 128
            :param res_blocks: 10
            :param hop_length: 275
            :param sample_rate: 22050
            :param mode: MOL or RAW
        '''
        super(WaveRNN, self).__init__()
        self.mode = mode
        self.pad = pad
        if self.mode == 'RAW':
            self.n_classes = 2 ** bits
        elif self.mode == 'MOL':
            self.n_classes = 30
        else:
            RuntimeError('Unknown model mode value - ', self.mode)

        # List of rnns to call 'flatten_parameters()' on
        self._to_flatten = []

        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4                               # 128 // 4 = 32
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.upsample = UpsampleNetwork(feat_dims, upsample_factors, compute_dims, res_blocks, res_out_dims, pad)
        self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)     # 80 + 32 + 1 = 113 -> 512

        self.rnn1 = nn.GRU(rnn_dims, rnn_dims)                          # 512 -> 512
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims)          # 512 + 32 = 544 -> 512
        self._to_flatten += [self.rnn1, self.rnn2]

        self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)         # 512 + 32 = 544 -> 512
        self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)          # 512 + 32 = 544 -> 512
        self.fc3 = nn.Linear(fc_dims, self.n_classes)                   # 512 -> 30

        self.num_params()

        # Avoid fragmentation of RNN parameters and associated warning
        self._flatten_parameters()


    def forward(self, x, mels):
        '''
        :param x: wav sequence, [batch_size, timeseq]
        :param mels: mel spectrogram [batch_size, n_mels, n_frames]
        :return:
        '''
        # Although we `_flatten_parameters()` on init, when using DataParallel
        # the model gets replicated, making it no longer guaranteed that the
        # weights are contiguous in GPU memory. Hence, we must call it again
        self._flatten_parameters()

        bsize = x.shape[0]
        h1 = paddle.zeros([1, bsize, self.rnn_dims])
        h2 = paddle.zeros([1, bsize, self.rnn_dims])
        # time_seq = n_frames * hop_length - 2 * pad * hop_length
        # mels: [batch_size, time_seq, n_mels], aux: [batch_size, time_seq, res_out_dims]
        mels, aux = self.upsample(mels)

        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]
        # x: [batch_size, time_seq] -> [batch_size, time_seq, 1+n_mels+aux_dims]
        x = paddle.concat([x.unsqueeze(-1), mels, a1], axis=2)
        x = self.I(x)
        res = x
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = paddle.concat([x, a2], axis=2)
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = paddle.concat([x, a3], axis=2)
        x = F.relu(self.fc1(x))

        x = paddle.concat([x, a4], axis=2)
        x = F.relu(self.fc2(x))

        return self.fc3(x)

    @paddle.no_grad()
    def generate(self, mels, batched, target, overlap, mu_law, gen_display=True):
        """
        Args:
            :param mels: input mels, [1, 80, n_frames]
            :param batched: generate in batch or not
            :param target: target number of samples to be generated in each batch entry
            :param overlap: number of samples for crossfading between batches
            :param mu_law: use mu law or not
        :return: wav sequence
        """

        self.eval()

        mu_law = mu_law if self.mode == 'RAW' else False

        output = []
        start = time.time()
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        wave_len = (mels.shape[-1] - 1) * self.hop_length
        # TODO remove two transpose op by modifying function pad_tenosr
        mels = self.pad_tensor(mels.transpose([0, 2, 1]), pad=self.pad, side='both')
        mels, aux = self.upsample(mels.transpose([0, 2, 1]))

        if batched:
            # (num_folds, target + 2 * overlap, features)
            mels = self.fold_with_overlap(mels, target, overlap)
            aux = self.fold_with_overlap(aux, target, overlap)

        b_size, seq_len, _ = mels.shape
        h1 = paddle.zeros([b_size, self.rnn_dims])
        h2 = paddle.zeros([b_size, self.rnn_dims])
        x = paddle.zeros([b_size, 1])

        d = self.aux_dims
        aux_split = [aux[:, :, d*i:d*(i + 1)] for i in range(4)]

        for i in range(seq_len):
            m_t = mels[:, i, :]

            a1_t, a2_t, a3_t, a4_t = (a[:, i, :] for a in aux_split)
            x = paddle.concat([x, m_t, a1_t], axis=1)
            x = self.I(x)
            h1, _ = rnn1(x, h1)
            x = x + h1
            inp = paddle.concat([x, a2_t], axis=1)
            h2, _ = rnn2(inp, h2)

            x = x + h2
            x = paddle.concat([x, a3_t], axis=1)
            x = F.relu(self.fc1(x))

            x = paddle.concat([x, a4_t], axis=1)
            x = F.relu(self.fc2(x))

            logits = self.fc3(x)

            if self.mode == 'MOL':
                sample = sample_from_discretized_mix_logistic(logits.unsqueeze(0).transpose([0, 2, 1]))
                output.append(sample.reshape([-1]))
                x = sample.transpose([1, 0, 2])

            elif self.mode == 'RAW':
                posterior = F.softmax(logits, axis=1)
                distrib = paddle.distribution.Categorical(posterior)
                # corresponding operate [np.floor((fx + 1) / 2 * mu + 0.5)] in enocde_mu_law
                sample = 2 * distrib.sample([1])[0].cast('float32') / (self.n_classes - 1.) - 1.
                output.append(sample)
                x = sample.unsqueeze(-1)
            else:
                raise RuntimeError('Unknown model mode value - ', self.mode)
            
            if gen_display:
                if i % 100 == 0:
                    self.gen_display(i, seq_len, b_size, start)

        output = paddle.stack(output).transpose([1, 0])
        output = output.cpu().numpy()
        output = output.astype(np.float64)

        if mu_law:
            output = decode_mu_law(output, self.n_classes, False)

        if batched:
            output = self.xfade_and_unfold(output, target, overlap)
        else:
            output = output[0]

        # Fade-out at the end to avoid signal cutting out suddenly
        fade_out = np.linspace(1, 0, 20 * self.hop_length)
        output = output[:wave_len]
        output[-20 * self.hop_length:] *= fade_out

        self.train()

        return output

    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh = gru.weight_hh_l0
        gru_cell.weight_ih = gru.weight_ih_l0
        gru_cell.bias_hh = gru.bias_hh_l0
        gru_cell.bias_ih = gru.bias_ih_l0

        return gru_cell


    def num_params(self, print_out=True):
        # calculate the num of trainable parameters of model
        parameters = filter(lambda p: not p.stop_gradient, self.parameters())
        parameters = sum([np.prod(p.shape) for p in parameters]) / 1000000
        if print_out:
            print('Trainable Parameters: {:.3f}M'.format(parameters))
        return parameters

    def _flatten_parameters(self):
        [m.flatten_parameters() for m in self._to_flatten]

    def pad_tensor(self, x, pad, side='both'):
        '''
        Args:
            :param x: mel, [1, n_frames, 80]
            :param pad:
            :param side: 'both', 'before' or 'after'
        :return:
        '''
        b, t, c = x.shape
        total = t + 2 * pad if side == 'both' else t + pad
        padded = paddle.zeros([b, total, c])
        if side == 'before' or side == 'both':
            padded[:, pad:pad+t, :] = x
        elif side == 'after':
            padded[:, :t, :] = x
        return padded

        '''
        if side not in ['both', 'before', 'after']:
            raise ValueError("'side' must be in ['both', 'before', 'after']!")
        left_pad = pad if side == 'both' or side == 'before' else 0
        right_pad = pad if side == 'both' or sode == 'after' else 0
        padded = nn.functional.pad(x, [0, 0, left_pad, right_pad], mode='constant', value=0)
        return padded
        '''

    def fold_with_overlap(self, x, target, overlap):

        '''
        Fold the tensor with overlap for quick batched inference.
        Overlap will be used for crossfading in xfade_and_unfold()

        Args:
            x (tensor)    : Upsampled conditioning features. mels or aux
                            shape=(1, timesteps, features)
                            mels: [1, time_seq, 80]
                            aux: [1, time_seq, 128]
            target (int)  : Target timesteps for each index of batch    11_000
            overlap (int) : Timesteps for both xfade and rnn warmup     550  overlap = hop_length * 2

        Return:
            (tensor) : shape=(num_folds, target + 2 * overlap, features)
            target + 2 * overlap = 12100
            num_flods = (time_seq - 550) // (11000 + 550)
            mel: [num_folds, 12100, 80]
            aux: [num_folds, 12100, 128]

        Details:
            x = [[h1, h2, ... hn]]

            Where each h is a vector of conditioning features

            Eg: target=2, overlap=1 with x.size(1)=10

            folded = [[h1, h2, h3, h4],
                      [h4, h5, h6, h7],
                      [h7, h8, h9, h10]]
        '''

        _, total_len, features = x.shape

        # Calculate variables needed
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, side='after')

        folded = paddle.zeros([num_folds, target + 2 * overlap, features])

        # Get the values for the folded tensor
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]
        return folded

    def xfade_and_unfold(self, y, target, overlap):
        ''' Applies a crossfade and unfolds into a 1d array.
        default:
            target = 11000
            overlap = 550
        Args:
            y (ndarry)    : Batched sequences of audio samples
                            shape=(num_folds, target + 2 * overlap)
                            dtype=np.float64
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (ndarry) : audio samples in a 1d array
                       shape=(total_len)
                       dtype=np.float64

        Details:
            y = [[seq1],
                 [seq2],
                 [seq3]]

            Apply a gain envelope at both ends of the sequences

            y = [[seq1_in, seq1_target, seq1_out],
                 [seq2_in, seq2_target, seq2_out],
                 [seq3_in, seq3_target, seq3_out]]

            Stagger and add up the groups of samples:

            [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]

        '''
        # num_folds = (total_len - overlap) // (target + overlap), length = 11000 + 2 * 550 = 12100
        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap

        # Need some silence for the run warmup
        slience_len = overlap // 2
        fade_len = overlap - slience_len
        slience = np.zeros([slience_len], dtype=np.float64)
        linear = np.ones([fade_len], dtype=np.float64)

        # Equal power crossfade
        # fade_in increase from 0 to 1, fade_out reduces from 1 to 0
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))
        # Concat the silence to the fades
        # fade_in: [overlap] [0, 0, ..., 0, 0.001 ,0.0011, ..., 0.98, 0.991, ..., 1.0] 275 zeros + 275 from 0 to 1
        # fade_out: [overlap] [1.0, 1.0, ..., 1.0, 0.999, 0.998, ..., 0.111, 0.110, ..., 0.0] 275 ones + 275 from 1 to 0
        fade_out = np.concatenate([linear, fade_out])
        fade_in = np.concatenate([slience, fade_in])

        # Apply the gain to the overlap samples
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out

        unfolded = np.zeros([total_len], dtype=np.float64)

        # Loop to add up all the samples
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]

        return unfolded


    def gen_display(self, i, seq_len, b_size, start):
        gen_rate = (i + 1) / (time.time() - start) * b_size / 1000
        pbar = self.progbar(i, seq_len)
        msg = f'| {pbar} {i*b_size}/{seq_len*b_size} | Batch Size: {b_size} | Gen Rate: {gen_rate:.1f}kHz | '
        sys.stdout.write(f"\r{msg}")


    def progbar(self, i, n, size=16):
        done = int(i * size) // n
        bar = ''
        for i in range(size):
            bar += '█' if i <= done else '░'
        return bar
    
    @classmethod
    def from_pretrained(cls, config, checkpoint_path):
        """Build a ConditionalWaveFlow model from a pretrained model.

        Parameters
        ----------
        config: yacs.config.CfgNode
            model configs

        checkpoint_path: Path or str
            the path of pretrained model checkpoint, without extension name

        Returns
        -------
        ConditionalWaveFlow
            The model built from pretrained result.
        """
        model = cls(rnn_dims=config.model.rnn_dims, fc_dims=config.model.fc_dims, bits=config.data.bits,
                    pad=config.model.pad, upsample_factors=config.model.upsample_factors,
                    feat_dims=config.data.num_mels, compute_dims=config.model.compute_dims,
                    res_out_dims=config.model.res_out_dims, res_blocks=config.model.res_blocks,
                    hop_length=config.data.hop_length, sample_rate=config.data.sample_rate,
                    mode=config.model.mode)
        checkpoint.load_parameters(model, checkpoint_path=checkpoint_path)
        return model


if __name__ == '__main__':
    from config import get_cfg_defaults
    config = get_cfg_defaults()

    wavernn = WaveRNN(rnn_dims=config.model.rnn_dims, fc_dims=config.model.fc_dims, bits=config.data.bits,
                      pad=config.model.pad, upsample_factors=config.model.upsample_factors,
                      feat_dims=config.data.num_mels, compute_dims=config.model.compute_dims,
                      res_out_dims=config.model.res_out_dims, res_blocks=config.model.res_blocks,
                      hop_length=config.data.hop_length, sample_rate=config.data.sample_rate,
                      mode=config.model.mode)


    batch_size = 4
    n_frames = 20 + 4
    time_len = 20 * config.data.hop_length

    x = paddle.rand([batch_size, time_len])
    # print(config.data.num_mels)
    mel = paddle.rand([batch_size, config.data.num_mels, n_frames])

    # out = wavernn(x, mel)
    # print(out.shape)

    # paddle.summary(wavernn, [(batch_size, time_len), (batch_size, config.data.num_mels, n_frames)])

    param_list = list(wavernn.state_dict().keys())
    for i, param in enumerate(param_list):
        print('{} {}'.format(i, param))

    # generate(self, mels, batched, target, overlap, mu_law)
    # output = wavernn.generate(paddle.rand([1, 80, 30]), True, 11000, 550, True)
    # print(output.shape)
















