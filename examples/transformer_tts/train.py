import time
import logging
from pathlib import Path
import numpy as np
import paddle
from paddle import distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler
from tensorboardX import SummaryWriter
from collections import defaultdict

import parakeet
from parakeet.data import dataset
from parakeet.frontend import English
from parakeet.models.transformer_tts import TransformerTTS, TransformerTTSLoss
from parakeet.utils import scheduler, checkpoint, mp_tools, display
from parakeet.training.cli import default_argument_parser
from parakeet.training.experiment import ExperimentBase

from config import get_cfg_defaults
from ljspeech import LJSpeech, LJSpeechCollector, Transform

class Experiment(ExperimentBase):
    def setup_model(self):
        config = self.config
        frontend = English()
        model = TransformerTTS(
            frontend, 
            d_encoder=config.model.d_encoder,
            d_decoder=config.model.d_decoder,
            d_mel=config.data.d_mel,
            n_heads=config.model.n_heads,
            d_ffn=config.model.d_ffn,
            encoder_layers=config.model.encoder_layers,
            decoder_layers=config.model.decoder_layers,
            d_prenet=config.model.d_prenet,
            d_postnet=config.model.d_postnet,
            postnet_layers=config.model.postnet_layers,
            postnet_kernel_size=config.model.postnet_kernel_size,
            max_reduction_factor=config.model.max_reduction_factor,
            decoder_prenet_dropout=config.model.decoder_prenet_dropout,
            dropout=config.model.dropout)
        if self.parallel:
            model = paddle.DataParallel(model)
        optimizer = paddle.optimizer.Adam(
            learning_rate=config.training.lr,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-9,
            parameters=model.parameters()
        )
        criterion = TransformerTTSLoss(config.model.stop_loss_scale)
        drop_n_heads = scheduler.StepWise(config.training.drop_n_heads)
        reduction_factor = scheduler.StepWise(config.training.reduction_factor)

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.drop_n_heads = drop_n_heads
        self.reduction_factor = reduction_factor

    def setup_dataloader(self):
        args = self.args
        config = self.config

        ljspeech_dataset = LJSpeech(args.data)
        transform = Transform(config.data.mel_start_value, config.data.mel_end_value)
        ljspeech_dataset = dataset.TransformDataset(ljspeech_dataset, transform)
        valid_set, train_set = dataset.split(ljspeech_dataset, config.data.valid_size)
        batch_fn = LJSpeechCollector(padding_idx=config.data.padding_idx)
        
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

        valid_loader = DataLoader(
            valid_set, batch_size=config.data.batch_size, collate_fn=batch_fn)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def compute_outputs(self, text, mel, stop_label):
        model_core = self.model._layers if self.parallel else self.model
        model_core.set_constants(
            self.reduction_factor(self.iteration), 
            self.drop_n_heads(self.iteration))

        # TODO(chenfeiyu): we can combine these 2 slices
        mel_input = mel[:,:-1, :]
        reduced_mel_input = mel_input[:, ::model_core.r, :]
        outputs = self.model(text, reduced_mel_input)
        return outputs

    def compute_losses(self, inputs, outputs):
        _, mel, stop_label = inputs
        mel_target = mel[:, 1:, :]
        stop_label_target = stop_label[:, 1:]

        mel_output = outputs["mel_output"]
        mel_intermediate = outputs["mel_intermediate"]
        stop_logits = outputs["stop_logits"]

        time_steps = mel_target.shape[1]
        losses = self.criterion(
            mel_output[:,:time_steps, :], 
            mel_intermediate[:,:time_steps, :], 
            mel_target, 
            stop_logits[:,:time_steps, :], 
            stop_label_target)
        return losses

    def train_batch(self):
        start = time.time()
        batch = self.read_batch()
        data_loader_time = time.time() - start

        self.optimizer.clear_grad()
        self.model.train()
        text, mel, stop_label = batch
        outputs = self.compute_outputs(text, mel, stop_label)
        losses = self.compute_losses(batch, outputs)
        loss = losses["loss"]
        loss.backward() 
        self.optimizer.step()
        iteration_time = time.time() - start

        losses_np = {k: float(v) for k, v in losses.items()}
        # logging
        msg = "Rank: {}, ".format(dist.get_rank())
        msg += "step: {}, ".format(self.iteration)
        msg += "time: {:>.3f}s/{:>.3f}s, ".format(data_loader_time, iteration_time)
        msg += ', '.join('{}: {:>.6f}'.format(k, v) for k, v in losses_np.items())
        self.logger.info(msg)
        
        if dist.get_rank() == 0:
            for k, v in losses_np.items():
                self.visualizer.add_scalar(f"train_loss/{k}", v, self.iteration)
    
    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def valid(self):
        valid_losses = defaultdict(list)
        for i, batch in enumerate(self.valid_loader):
            text, mel, stop_label = batch
            outputs = self.compute_outputs(text, mel, stop_label)
            losses = self.compute_losses(batch, outputs)
            for k, v in losses.items():
                valid_losses[k].append(float(v))

            if i < 2:
                attention_weights = outputs["cross_attention_weights"]
                display.add_multi_attention_plots(
                    self.visualizer, 
                    f"valid_sentence_{i}_cross_attention_weights", 
                    attention_weights, 
                    self.iteration)

        # write visual log
        valid_losses = {k: np.mean(v) for k, v in valid_losses.items()}
        for k, v in valid_losses.items():
            self.visualizer.add_scalar(f"valid/{k}", v, self.iteration)


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
