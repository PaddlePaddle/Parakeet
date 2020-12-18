import time
from pathlib import Path
import math
import numpy as np
import paddle
from paddle import distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler
from tensorboardX import SummaryWriter
from collections import defaultdict

import parakeet
from parakeet.data import dataset
from parakeet.models.wavenet import UpsampleNet, WaveNet, ConditionalWaveNet
from parakeet.audio import AudioProcessor
from parakeet.utils import scheduler, mp_tools
from parakeet.training.cli import default_argument_parser
from parakeet.training.experiment import ExperimentBase
from parakeet.utils.mp_tools import rank_zero_only

from config import get_cfg_defaults
from ljspeech import LJSpeech, LJSpeechClipCollector, LJSpeechCollector


class Experiment(ExperimentBase):
    def setup_model(self):
        config = self.config
        model = ConditionalWaveNet(
            upsample_factors=config.model.upsample_factors,
            n_stack=config.model.n_stack, 
            n_loop=config.model.n_loop,
            residual_channels=config.model.residual_channels,
            output_dim=config.model.output_dim,
            n_mels=config.data.n_mels,
            filter_size=config.model.filter_size,
            loss_type=config.model.loss_type,
            log_scale_min=config.model.log_scale_min)

        if self.parallel > 1:
            model = paddle.DataParallel(model)

        lr_scheduler = paddle.optimizer.lr.StepDecay(
            config.training.lr, 
            config.training.anneal_interval, 
            config.training.anneal_rate)
        optimizer = paddle.optimizer.Adam(
            lr_scheduler,
            parameters=model.parameters(),
            grad_clip=paddle.nn.ClipGradByGlobalNorm(config.training.gradient_max_norm))

        self.model = model
        self.model_core = model._layer if self.parallel else model
        self.optimizer = optimizer

    def setup_dataloader(self):
        config = self.config
        args = self.args

        ljspeech_dataset = LJSpeech(args.data)
        valid_set, train_set = dataset.split(ljspeech_dataset, config.data.valid_size)

        # convolutional net's causal padding size
        context_size = config.model.n_stack \
                      * sum([(config.model.filter_size - 1) * 2**i for i in range(config.model.n_loop)]) \
                      + 1
        context_frames = context_size // config.data.hop_length

        # frames used to compute loss
        frames_per_second = config.data.sample_rate // config.data.hop_length
        train_clip_frames = math.ceil(config.data.train_clip_seconds * frames_per_second)
        
        num_frames = train_clip_frames + context_frames
        batch_fn = LJSpeechClipCollector(num_frames, config.data.hop_length)
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
                shuffle=True,
                drop_last=True)
            train_loader = DataLoader(
                train_set, batch_sampler=sampler, collate_fn=batch_fn)

        valid_batch_fn = LJSpeechCollector()
        valid_loader = DataLoader(
            valid_set, batch_size=1, collate_fn=valid_batch_fn)
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def train_batch(self):
        start = time.time()
        batch = self.read_batch()
        data_loader_time = time.time() - start

        self.model.train()
        self.optimizer.clear_grad()
        mel, wav, audio_starts = batch
        
        y = self.model(wav, mel, audio_starts)
        loss = self.model.loss(y, wav)
        loss.backward() 
        self.optimizer.step()
        iteration_time = time.time() - start

        loss_value = float(loss)
        msg = "Rank: {}, ".format(dist.get_rank())
        msg += "step: {}, ".format(self.iteration)
        msg += "time: {:>.3f}s/{:>.3f}s, ".format(data_loader_time, iteration_time)
        msg += "loss: {:>.6f}".format(loss_value)
        self.logger.info(msg)
        self.visualizer.add_scalar("train/loss", loss_value, global_step=self.iteration)

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def valid(self):
        valid_iterator = iter(self.valid_loader)
        valid_losses = []
        mel, wav, audio_starts = next(valid_iterator)
        y = self.model(wav, mel, audio_starts)
        loss = self.model.loss(y, wav)
        valid_losses.append(float(loss))
        valid_loss = np.mean(valid_losses)
        self.visualizer.add_scalar("valid/loss", valid_loss, global_step=self.iteration)


def main_sp(config, args):
    exp = Experiment(config, args)
    exp.setup()
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
