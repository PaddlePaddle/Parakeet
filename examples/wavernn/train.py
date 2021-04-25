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

import time
import numpy as np
import soundfile as sf
import paddle
from paddle import distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler
from tensorboardX import SummaryWriter
from collections import defaultdict

from parakeet.data import dataset
from parakeet.audio import AudioProcessor
from parakeet.utils import scheduler, mp_tools
from parakeet.training.cli import default_argument_parser
from parakeet.training.experiment import ExperimentBase
from parakeet.utils.mp_tools import rank_zero_only

from parakeet.models.wavernn import WaveRNN
from utils.audio import decode_mu_law, label_2_float
from utils.distribution import discretized_mix_logistic_loss
from config import get_cfg_defaults
from ljspeech import LJSpeech, LJSpeechCollate


def calculate_grad_norm(parameters, norm_type=2):
    '''
    calculate grad norm of mdoel's parameters
    :param parameters: model's parameters
    :param norm_type: 
    :return: grad_norm
    '''

    grad_list = [paddle.to_tensor(p.grad) for p in parameters if p.grad is not None]
    norm_list = paddle.stack([paddle.norm(grad, norm_type) for grad in grad_list])
    total_norm = paddle.norm(norm_list)
    return total_norm



class Experiment(ExperimentBase):
    def setup_model(self):
        config = self.config
        model = WaveRNN(rnn_dims=config.model.rnn_dims, fc_dims=config.model.fc_dims, bits=config.data.bits,
                        pad=config.model.pad, upsample_factors=config.model.upsample_factors,
                        feat_dims=config.data.num_mels, compute_dims=config.model.compute_dims,
                        res_out_dims=config.model.res_out_dims, res_blocks=config.model.res_blocks,
                        hop_length=config.data.hop_length, sample_rate=config.data.sample_rate,
                        mode=config.model.mode)
        if self.parallel:
            model = paddle.DataParallel(model)
        clip = paddle.nn.ClipGradByGlobalNorm(self.config.training.clip_grad_norm)
        optimizer = paddle.optimizer.Adam(parameters=model.parameters(),
                                          learning_rate=config.training.lr, grad_clip=clip)

        if config.model.mode == 'RAW':
            criterion = paddle.nn.CrossEntropyLoss(axis=1)
        elif config.model.mode == 'MOL':
            criterion = discretized_mix_logistic_loss
        else:
            criterion = None
            RuntimeError('Unknown model mode value - ', config.model.mode)

        # create dir that generates valid samples during training
        self.setup_valid_samples()

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.mode = config.model.mode

    def setup_dataloader(self):
        config = self.config
        args = self.args

        ljspeech_dataset = LJSpeech(args.data)

        train_set, valid_set = dataset.split(ljspeech_dataset,
                                             len(ljspeech_dataset) - config.data.valid_size)

        batch_fn = LJSpeechCollate(mode=config.model.mode, seq_len=config.training.seq_len,
                                   hop_length=config.data.hop_length, pad=config.model.pad,
                                   bits=config.data.bits)

        if not self.parallel:
            train_loader = DataLoader(
                train_set,
                batch_size=config.data.batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=batch_fn)
        else:
            sampler = DistributedBatchSampler(
                train_set,
                batch_size=config.data.batch_size,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                drop_last=True)
            train_loader = DataLoader(
                train_set, batch_sampler=sampler, collate_fn=batch_fn)

        valid_loader = DataLoader(valid_set, collate_fn=batch_fn, batch_size=1)
        valid_generate_loader = DataLoader(valid_set, batch_size=1)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.valid_generate_loader = valid_generate_loader

    def train_batch(self):
        start = time.time()
        batch = self.read_batch()
        data_loader_time = time.time() - start

        self.model.train()
        self.optimizer.clear_grad()
        wav, y, mel = batch
        y_hat = self.model(wav, mel)
        if self.mode == 'RAW':
            y_hat = y_hat.transpose([0, 2, 1]).unsqueeze(-1)
        elif self.mode == 'MOL':
            y_hat = paddle.cast(y, dtype='float32')

        y = y.unsqueeze(-1)
        loss = self.criterion(y_hat, y)
        loss.backward()

        grad_norm = float(calculate_grad_norm(self.model.parameters(), norm_type=2))

        self.optimizer.step()
        iteration_time = time.time() - start

        loss_value = float(loss)
        msg = "Rank: {}, ".format(dist.get_rank())
        msg += "step: {}, ".format(self.iteration)
        msg += "time: {:>.3f}s/{:>.3f}, ".format(data_loader_time, iteration_time)
        msg += "loss: {:>.6f}, ".format(loss_value)
        msg += "gradnorm: {:>.6f}".format(grad_norm)
        self.logger.info(msg)
        if np.isnan(grad_norm):
            print('grad_norm was NaN!')
        self.visualizer.add_scalar("train/loss", loss_value, global_step=self.iteration)
        self.visualizer.add_scalar("train/grad_norm", grad_norm, global_step=self.iteration)

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def valid(self):
        if self.iteration % self.config.training.valid_generate_valid_interval == 0:
            self.gen_valid_samples()

        valid_losses = []
        for wav, y, mel in self.valid_loader:
            y_hat = self.model(wav, mel)
            if self.mode == 'RAW':
                y_hat = y_hat.transpose([0, 2, 1]).unsqueeze(-1)
            elif self.mode == 'MOL':
                y_hat = paddle.cast(y, dtype='float32')

            y = y.unsqueeze(-1)
            loss = self.criterion(y_hat, y)

            valid_losses.append(float(loss))
        valid_loss = np.mean(valid_losses)
        self.visualizer.add_scalar(
            "valid/loss", valid_loss, global_step=self.iteration)

    def gen_valid_samples(self):
        for i, (mel, wav) in enumerate(self.valid_generate_loader):
            if i > self.config.training.generate_at_checkpoint:
                break
            print('\n| Generating: {}/{}'.format(i, self.config.training.generate_at_checkpoint))
            wav = wav[0].numpy()
            if self.mode == 'MOL':
                bits = 16
            else:
                bits = self.config.data.bits
            if self.config.data.mu_law and self.mode != 'MOL':
                wav = decode_mu_law(wav, 2**bits, from_labels=True)
            else:
                wav = label_2_float(wav, bits)
            origin_save_path = self.valid_samples_dir / '{}k_steps_{}_target.wav'.format(
                self.iteration//1000, i
            )
            sf.write(origin_save_path, wav, samplerate=self.config.data.sample_rate)

            if self.config.model.gen_batched:
                batch_str = 'gen_batched_target{}_overlap{}'.format(
                    self.config.model.target, self.config.model.overlap
                )
            else:
                batch_str = 'gen_not_batched'
            gen_save_path = str(self.valid_samples_dir / '{}k_steps_{}_{}.wav'.format(
                self.iteration//1000, i, batch_str)
            )
            gen_sample = self.model.generate(
                mel, self.config.model.gen_batched, self.config.model.target,
                self.config.model.overlap, self.config.data.mu_law)
            sf.write(gen_save_path, gen_sample, samplerate=self.config.data.sample_rate)

    @mp_tools.rank_zero_only
    def setup_valid_samples(self):
        """Create a directory used to save checkpoints into.

        It is "checkpoints" inside the output directory.
        """
        valid_samples_dir = self.output_dir / 'valid_samples'
        valid_samples_dir.mkdir(exist_ok=True)

        self.valid_samples_dir = valid_samples_dir


def main_sp(config, args):
    exp = Experiment(config, args)
    exp.setup()
    # exp.valid()
    exp.run()


def main(config, args):
    if args.nprocs > 1 and args.device == "gpu":
        dist.spawn(main_sp, args=(config, args), nprocs=args.nprocs)
    else:
        main_sp(config, args)


if __name__ == "__main__":
    config = get_cfg_defaults()
    parser = default_argument_parser()
    args = parser.parse_args()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config)
    print(args)

    main(config, args)






