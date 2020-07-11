import numpy as np 
from matplotlib import cm
import librosa
import os
import time
import tqdm
import argparse
from ruamel import yaml
import paddle
from paddle import fluid
from paddle.fluid import layers as F
from paddle.fluid import dygraph as dg
from paddle.fluid.io import DataLoader
from tensorboardX import SummaryWriter
import soundfile as sf

from parakeet.data import SliceDataset, DataCargo, PartialyRandomizedSimilarTimeLengthSampler, SequentialSampler
from parakeet.utils.io import save_parameters, load_parameters, add_yaml_config_to_args
from parakeet.g2p import en

from vocoder import WaveflowVocoder
from train import create_model


def main(args, config):
    model = create_model(config)
    loaded_step = load_parameters(model, checkpoint_path=args.checkpoint)
    model.eval()
    vocoder = WaveflowVocoder()
    vocoder.model.eval()
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    monotonic_layers = [int(item.strip()) - 1 for item in args.monotonic_layers.split(',')]
    with open(args.input, 'rt') as f:
        sentences = [line.strip() for line in f.readlines()]
    for i, sentence in enumerate(sentences):
        wav = synthesize(config, model, vocoder, sentence, monotonic_layers)
        sf.write(os.path.join(args.output, "sentence{}.wav".format(i)),
                 wav, samplerate=config["sample_rate"])


def synthesize(config, model, vocoder, sentence, monotonic_layers):
    print("[synthesize] {}".format(sentence))
    text = en.text_to_sequence(sentence, p=1.0)
    text = np.expand_dims(np.array(text, dtype="int64"), 0)
    lengths = np.array([text.size], dtype=np.int64)
    text_seqs = dg.to_variable(text)
    text_lengths = dg.to_variable(lengths)

    decoder_layers = config["decoder_layers"]
    force_monotonic_attention = [False] * decoder_layers
    for i in monotonic_layers:
        force_monotonic_attention[i] = True
    
    with dg.no_grad():
        outputs = model(text_seqs, text_lengths, speakers=None,
            force_monotonic_attention=force_monotonic_attention, 
            window=(config["backward_step"], config["forward_step"]))
        decoded, refined, attentions = outputs
        wav = vocoder(F.transpose(decoded, (0, 2, 1)))
        wav_np = wav.numpy()[0]
    return wav_np


if __name__ == "__main__":
    import argparse
    from ruamel import yaml
    parser = argparse.ArgumentParser("synthesize from a checkpoint")
    parser.add_argument("--config", type=str, required=True, help="config file")
    parser.add_argument("--input", type=str, required=True, help="text file to synthesize")
    parser.add_argument("--output", type=str, required=True, help="path to save audio")
    parser.add_argument("--checkpoint", type=str, required=True, help="data path of the checkpoint")
    parser.add_argument("--monotonic_layers", type=str, required=True, help="monotonic decoder layer, index starts friom 1")
    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        config = yaml.safe_load(f)
    
    dg.enable_dygraph(fluid.CUDAPlace(0))
    main(args, config)