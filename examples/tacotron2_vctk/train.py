import time
from collections import defaultdict
import numpy as np
import librosa

import paddle
from paddle import distributed as dist
from paddle import DataParallel
from paddle.io import DataLoader, DistributedBatchSampler
from paddle.optimizer import Adam

import parakeet
from parakeet.data import dataset
from parakeet.frontend import EnglishCharacter
from parakeet.training.cli import default_argument_parser
from parakeet.training.experiment import ExperimentBase
from parakeet.utils import display, mp_tools
from parakeet.models.tacotron2 import Tacotron2, Tacotron2Loss

from config import get_cfg_defaults
from vctk import VCTK, collate_vctk_examples


class TacotronVCTKExperiment(ExperimentBase):
    def setup_model(self):
        config = self.config
        model_config = config.model
        data_config = config.data

        model = Tacotron2(
            vocab_size=model_config.vocab_size,
            num_speakers=model_config.num_speakers,
            d_speaker=model_config.d_speaker,
            d_mels=data_config.d_mels,
            d_encoder=model_config.d_encoder,
            encoder_conv_layers=model_config.encoder_conv_layers,
            encoder_kernel_size=model_config.encoder_kernel_size,
            d_prenet=model_config.d_prenet,
            d_attention_rnn=model_config.d_attention_rnn,
            d_decoder_rnn=model_config.d_decoder_rnn,
            attention_filters=model_config.attention_filters,
            attention_kernel_size=model_config.attention_kernel_size,
            d_attention=model_config.d_attention,
            d_postnet=model_config.d_postnet,
            postnet_kernel_size=model_config.postnet_kernel_size,
            postnet_conv_layers=model_config.postnet_conv_layers,
            reduction_factor=model_config.reduction_factor,
            p_encoder_dropout=model_config.p_encodewr_dropout,
            p_prenet_dropout=model_config.p_prenet_dropout,
            p_attention_dropout=model_config.p_attention_dropout,
            p_decoder_dropout=model_config.p_decoder_dropout,
            p_postnet_dropout=model_config.p_postnet_dropout)
        self.model_core = model
        self.model = DataParallel(model) if self.parallel else model

        grad_clip = paddle.nn.ClipGradByGlobalNorm(
            config.training.grad_clip_thresh)
        optimizer = Adam(learning_rate=config.training.lr,
                         parameters=model.parameters(),
                         weight_decay=paddle.regularizer.L2Decay(
                             config.training.weight_decay),
                         grad_clip=grad_clip)
        self.optimizer = optimizer

        criterion = Tacotron2Loss(config.mode.guided_attn_loss_sigma)
        self.criterion = criterion

    def setup_dataloader(self):
        config = self.config
        args = self.args

        dataset = VCTK(args.data)
        valid_dataset, train_dataset = dataset.split(dataste,
                                                     config.data.valid_size)
        if self.parallel:
            sampler = DistributedBatchSampler(
                train_dataset,
                batch_size=config.data.batch_size,
                shuffle=True,
                drop_last=True)
            self.train_loader = DataLoader(train_dataset,
                                           batch_sampler=sampler,
                                           collate_fn=collate_vctk_examples,
                                           num_workers=4)
        else:
            self.train_loader = DataLoader(train_dataset,
                                           batch_size=config.data.batch_size,
                                           num_workers=8,
                                           shuffle=True,
                                           drop_last=True)
        self.valid_loader = DataLoader(valid_dataset, 
                                       batch_size=1,
                                       num_workers=1,
                                       shuffle=False, 
                                       drop_last=False)
        

    def train_batch(self):
        if self.parallel:
            dist.barrier()
            
        start = time.time()
        batch = self.read_batch()
        data_loader_time = time.time() - start
    
        self.optimizer.clear_grad()
        self.model.train()
        phonemes, plens, mels, slens, speaker_ids = batch
        
        outputs = self.model(phonemes, plens, mels, slens, speaker_ids)
        
        losses = self.criterion(outputs["mel_output"], 
                                outputs["mel_outputs_postnets"],
                                mels, 
                                outputs["alignments"],
                                slens,
                                plens)
        loss = losses["loss"]
        loss.backward()
        self.optimizer.step()
        iteration_time = time.time() - start
    
        losses_np = {k: float(v) for k, v in losses.items()}
        # logging
        msg = "Rank: {}, ".format(dist.get_rank())
        msg += "step: {}, ".format(self.iteration)
        msg += "time: {:>.3f}s/{:>.3f}s, ".format(data_loader_time,
                                                      iteration_time)
        msg += ', '.join('{}: {:>.6f}'.format(k, v)
                             for k, v in losses_np.items())
        self.logger.info(msg)
    
        if dist.get_rank() == 0:
            for k, v in losses_np.items():
                self.visualizer.add_scalar(f"train_loss/{k}", v,
                                               self.iteration)  
                
    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def valid(self):
        # this is evaluation 
        self.model.eval()
        model_core = self.model_core
        for i, batch in self.valid_loader:
            phonemes, plens, mels, slens, speaker_ids = batch
            outputs = model_core.infer(phonemes, speaker_ids=speaker_ids)
            
            fig = display.plot_spectrogram(output["mel_outputs_postnet"][0].numpy().T)
            self.visualizer.add_figure(f"sentence_{i}/predicted_mel", fig, self.iteration)
            
            fig = display.plot_spectrogram(mels[0].numpy().T)
            self.visualizer.add_figure(f"sentence_{i}/ground_truth_mel", fig, self.iteration)
            
            fig = display.plot_alignment(outputs["alignments"][0].numpy())
            self.visualizer.add_figure(f"sentence_{i}/predicted_mel", fig, self.iteration)
            
            mel_basis = librosa.filters.mel(22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)
            _inv_mel_basis = np.linalg.pinv(mel_basis)
            spec = np.matmul(_inv_mel_basis, np.exp(output["mel_outputs_postnet"][0].numpy().T))
            wav = librosa.core.griffinlim(spec, hop_length=256, win_length=1024)
            self.visualizer.add_audio(f"predicted/sentence_{i}", wav, self.iteration, sample_rate=22050)

        

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
