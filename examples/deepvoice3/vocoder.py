import argparse
from ruamel import yaml
import numpy as np
import librosa
import paddle
from paddle import fluid
from paddle.fluid import layers as F
from paddle.fluid import dygraph as dg
from parakeet.utils.io import load_parameters
from parakeet.models.waveflow.waveflow_modules import WaveFlowModule

class WaveflowVocoder(object):
    def __init__(self):
        config_path = "waveflow_res128_ljspeech_ckpt_1.0/waveflow_ljspeech.yaml"
        with open(config_path, 'rt') as f:
           config = yaml.safe_load(f)
        ns = argparse.Namespace()
        for k, v in config.items():
            setattr(ns, k, v)
        ns.use_fp16 = False
        
        self.model = WaveFlowModule(ns)
        checkpoint_path = "waveflow_res128_ljspeech_ckpt_1.0/step-2000000"
        load_parameters(self.model, checkpoint_path=checkpoint_path)

    def __call__(self, mel):
        with dg.no_grad():
            self.model.eval()
            audio = self.model.synthesize(mel)
        self.model.train()
        return audio

class GriffinLimVocoder(object):
    def __init__(self, sharpening_factor=1.4, sample_rate=22050, n_fft=1024, 
                 win_length=1024, hop_length=256):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.sharpening_factor = sharpening_factor
        self.win_length = win_length
        self.hop_length = hop_length

    def __call__(self, mel):
        spec = librosa.feature.inverse.mel_to_stft(
            np.exp(mel),
            sr=self.sample_rate,
            n_fft=self.n_fft,
            fmin=0, fmax=8000.0, power=1.0)
        audio = librosa.core.griffinlim(spec ** self.sharpening_factor, 
            win_length=self.win_length, hop_length=self.hop_length)
        return audio

