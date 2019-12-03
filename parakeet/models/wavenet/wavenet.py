import itertools
import os
import time

import librosa
import numpy as np
import paddle.fluid.dygraph as dg
from paddle import fluid

import utils
from data import LJSpeech
from wavenet_modules import WaveNetModule


class WaveNet():
    def __init__(self, config, checkpoint_dir, parallel=False, rank=0,
                 nranks=1, tb_logger=None):
        # Process config to calculate the context size
        dilations = list(
            itertools.islice(
                itertools.cycle(config.dilation_block), config.layers))
        config.context_size = sum(dilations) + 1
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.parallel = parallel
        self.rank = rank
        self.nranks = nranks
        self.tb_logger = tb_logger

    def build(self, training=True):
        config = self.config
        dataset = LJSpeech(config, self.nranks, self.rank) 
        self.trainloader = dataset.trainloader
        self.validloader = dataset.validloader

        wavenet = WaveNetModule("wavenet", config, self.rank)
        
        # Dry run once to create and initalize all necessary parameters.
        audio = dg.to_variable(np.random.randn(1, 20000).astype(np.float32))
        mel = dg.to_variable(
            np.random.randn(1, 100, self.config.mel_bands).astype(np.float32))
        audio_start = dg.to_variable(np.array([0], dtype=np.int32))
        wavenet(audio, mel, audio_start)

        if training:
            # Create Learning rate scheduler.
            lr_scheduler = dg.ExponentialDecay(
                learning_rate = config.learning_rate,
                decay_steps = config.anneal.every,
                decay_rate = config.anneal.rate,
                staircase=True)
    
            optimizer = fluid.optimizer.AdamOptimizer(
                learning_rate=lr_scheduler)
    
            clipper = fluid.dygraph_grad_clip.GradClipByGlobalNorm(
                config.gradient_max_norm)

            # Load parameters.
            utils.load_parameters(self.checkpoint_dir, self.rank,
                                  wavenet, optimizer,
                                  iteration=config.iteration,
                                  file_path=config.checkpoint)
            print("Rank {}: checkpoint loaded.".format(self.rank))
    
            # Data parallelism.
            if self.parallel:
                strategy = dg.parallel.prepare_context()
                wavenet = dg.parallel.DataParallel(wavenet, strategy)
    
            self.wavenet = wavenet
            self.optimizer = optimizer
            self.clipper = clipper

        else:
            # Load parameters.
            utils.load_parameters(self.checkpoint_dir, self.rank, wavenet,
                                  iteration=config.iteration,
                                  file_path=config.checkpoint)
            print("Rank {}: checkpoint loaded.".format(self.rank))

            self.wavenet = wavenet

    def train_step(self, iteration):
        self.wavenet.train()

        start_time = time.time()
        audios, mels, audio_starts = next(self.trainloader)
        load_time = time.time()

        loss, _ = self.wavenet(audios, mels, audio_starts)

        if self.parallel:
            # loss = loss / num_trainers
            loss = self.wavenet.scale_loss(loss)
            loss.backward()
            self.wavenet.apply_collective_grads()
        else:
            loss.backward()

        if isinstance(self.optimizer._learning_rate,
                      fluid.optimizer.LearningRateDecay):
            current_lr = self.optimizer._learning_rate.step().numpy()
        else:
            current_lr = self.optimizer._learning_rate

        self.optimizer.minimize(loss, grad_clip=self.clipper,
            parameter_list=self.wavenet.parameters())
        self.wavenet.clear_gradients()

        graph_time = time.time()

        if self.rank == 0:
            loss_val = float(loss.numpy()) * self.nranks
            log = "Rank: {} Step: {:^8d} Loss: {:<8.3f} " \
                  "Time: {:.3f}/{:.3f}".format(
                  self.rank, iteration, loss_val,
                  load_time - start_time, graph_time - load_time)
            print(log)

            tb = self.tb_logger
            tb.add_scalar("Train-Loss-Rank-0", loss_val, iteration)
            tb.add_scalar("Learning-Rate", current_lr, iteration)

    @dg.no_grad
    def valid_step(self, iteration):
        self.wavenet.eval()

        total_loss = []
        sample_audios = []
        start_time = time.time()
        for audios, mels, audio_starts in self.validloader():
            loss, sample_audio = self.wavenet(audios, mels, audio_starts, True)
            total_loss.append(float(loss.numpy()))
            sample_audios.append(sample_audio)
        total_time = time.time() - start_time

        if self.rank == 0:
            loss_val = np.mean(total_loss)
            log = "Test | Rank: {} AvgLoss: {:<8.3f} Time {:<8.3f}".format(
                self.rank, loss_val, total_time)
            print(log)

            tb = self.tb_logger
            tb.add_scalar("Valid-Avg-Loss", loss_val, iteration)
            tb.add_audio("Teacher-Forced-Audio-0", sample_audios[0].numpy(),
                iteration, sample_rate=self.config.sample_rate)
            tb.add_audio("Teacher-Forced-Audio-1", sample_audios[1].numpy(),
                iteration, sample_rate=self.config.sample_rate)

    @dg.no_grad
    def infer(self, iteration):
        self.wavenet.eval()

        config = self.config
        sample = config.sample

        output = "{}/{}/iter-{}".format(config.output, config.name, iteration)
        os.makedirs(output, exist_ok=True)

        filename = "{}/valid_{}.wav".format(output, sample)
        print("Synthesize sample {}, save as {}".format(sample, filename))

        mels_list = [mels for _, mels, _ in self.validloader()]
        start_time = time.time()
        syn_audio = self.wavenet.synthesize(mels_list[sample])
        syn_time = time.time() - start_time
        print("audio shape {}, synthesis time {}".format(
            syn_audio.shape, syn_time))
        librosa.output.write_wav(filename, syn_audio,
            sr=config.sample_rate)

    def save(self, iteration):
        utils.save_latest_parameters(self.checkpoint_dir, iteration,
                                     self.wavenet, self.optimizer)
        utils.save_latest_checkpoint(self.checkpoint_dir, iteration)
