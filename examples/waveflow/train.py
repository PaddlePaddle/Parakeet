import time
from pathlib import Path
import numpy as np
import paddle
from paddle import distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler
from tensorboardX import SummaryWriter
from collections import defaultdict

import parakeet
from parakeet.data import dataset
from parakeet.models.waveflow import UpsampleNet, WaveFlow, ConditionalWaveFlow, WaveFlowLoss
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
        model = ConditionalWaveFlow(
            upsample_factors=config.model.upsample_factors,
            n_flows=config.model.n_flows,
            n_layers=config.model.n_layers,
            n_group=config.model.n_group,
            channels=config.model.channels,
            n_mels=config.data.n_mels,
            kernel_size=config.model.kernel_size)

        if self.parallel > 1:
            model = paddle.DataParallel(model)
        optimizer = paddle.optimizer.Adam(config.training.lr, parameters=model.parameters())
        criterion = WaveFlowLoss(sigma=config.model.sigma)

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def setup_dataloader(self):
        config = self.config
        args = self.args

        ljspeech_dataset = LJSpeech(args.data)
        valid_set, train_set = dataset.split(ljspeech_dataset, config.data.valid_size)

        batch_fn = LJSpeechClipCollector(config.data.clip_frames, config.data.hop_length)
        
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

        valid_batch_fn = LJSpeechCollector()
        valid_loader = DataLoader(
            valid_set, batch_size=1, collate_fn=valid_batch_fn)
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def compute_outputs(self, mel, wav):
        # model_core = model._layers if isinstance(model, paddle.DataParallel) else model
        z, log_det_jocobian = self.model(wav, mel)
        return z, log_det_jocobian

    def compute_losses(self, outputs):
        loss = self.criterion(outputs)
        return loss

    def train_batch(self):
        start = time.time()
        batch = self.read_batch()
        data_loader_time = time.time() - start

        self.model.train()
        self.optimizer.clear_grad()
        mel, wav = batch
        outputs = self.compute_outputs(mel, wav)
        loss = self.compute_losses(outputs)
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
        mel, wav = next(valid_iterator)
        outputs = self.compute_outputs(mel, wav)
        loss = self.compute_losses(outputs)
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
