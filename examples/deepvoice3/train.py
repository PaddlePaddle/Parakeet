import numpy as np 
from matplotlib import cm
import librosa
import os
import time
import tqdm
import paddle
from paddle import fluid
from paddle.fluid import layers as F
from paddle.fluid import initializer as I
from paddle.fluid import dygraph as dg
from paddle.fluid.io import DataLoader
from tensorboardX import SummaryWriter

from parakeet.models.deepvoice3 import Encoder, Decoder, PostNet, SpectraNet
from parakeet.data import SliceDataset, DataCargo, SequentialSampler, RandomSampler
from parakeet.utils.io import save_parameters, load_parameters
from parakeet.g2p import en

from data import LJSpeech, DataCollector
from vocoder import WaveflowVocoder, GriffinLimVocoder
from clip import DoubleClip


def create_model(config):
    char_embedding = dg.Embedding((en.n_vocab, config["char_dim"]), param_attr=I.Normal(scale=0.1))
    multi_speaker = config["n_speakers"] > 1
    speaker_embedding = dg.Embedding((config["n_speakers"], config["speaker_dim"]), param_attr=I.Normal(scale=0.1)) \
        if multi_speaker else None
    encoder = Encoder(config["encoder_layers"], config["char_dim"], 
                      config["encoder_dim"], config["kernel_size"], 
                      has_bias=multi_speaker, bias_dim=config["speaker_dim"], 
                      keep_prob=1.0 - config["dropout"])
    decoder = Decoder(config["n_mels"], config["reduction_factor"], 
                      list(config["prenet_sizes"]) + [config["char_dim"]], 
                      config["decoder_layers"], config["kernel_size"], 
                      config["attention_dim"],
                      position_encoding_weight=config["position_weight"], 
                      omega=config["position_rate"], 
                      has_bias=multi_speaker, bias_dim=config["speaker_dim"], 
                      keep_prob=1.0 - config["dropout"])
    postnet = PostNet(config["postnet_layers"], config["char_dim"], 
                      config["postnet_dim"], config["kernel_size"], 
                      config["n_mels"], config["reduction_factor"], 
                      has_bias=multi_speaker, bias_dim=config["speaker_dim"], 
                      keep_prob=1.0 - config["dropout"])
    spectranet = SpectraNet(char_embedding, speaker_embedding, encoder, decoder, postnet)
    return spectranet

def create_data(config, data_path):
    dataset = LJSpeech(data_path)

    train_dataset = SliceDataset(dataset, config["valid_size"], len(dataset))
    train_collator = DataCollector(config["p_pronunciation"])
    train_sampler = RandomSampler(train_dataset)
    train_cargo = DataCargo(train_dataset, train_collator, 
        batch_size=config["batch_size"], sampler=train_sampler)
    train_loader = DataLoader\
                 .from_generator(capacity=10, return_list=True)\
                 .set_batch_generator(train_cargo)

    valid_dataset = SliceDataset(dataset, 0, config["valid_size"])
    valid_collector = DataCollector(1.)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_cargo = DataCargo(valid_dataset, valid_collector, 
        batch_size=1, sampler=valid_sampler)
    valid_loader = DataLoader\
                 .from_generator(capacity=2, return_list=True)\
                 .set_batch_generator(valid_cargo)
    return train_loader, valid_loader

def create_optimizer(model, config):
    optim = fluid.optimizer.Adam(config["learning_rate"], 
        parameter_list=model.parameters(), 
        grad_clip=DoubleClip(config["clip_value"], config["clip_norm"]))
    return optim

def train(args, config):
    model = create_model(config)
    train_loader, valid_loader = create_data(config, args.input)
    optim = create_optimizer(model, config)

    global global_step
    max_iteration = config["max_iteration"]
    
    iterator = iter(tqdm.tqdm(train_loader))
    while global_step <= max_iteration:
        # get inputs
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(tqdm.tqdm(train_loader))
            batch = next(iterator)
        
        # unzip it
        text_seqs, text_lengths, specs, mels, num_frames = batch

        # forward & backward
        model.train()
        outputs = model(text_seqs, text_lengths, speakers=None, mel=mels)
        decoded, refined, attentions, final_state = outputs

        causal_mel_loss = model.spec_loss(decoded, mels, num_frames)
        non_causal_mel_loss = model.spec_loss(refined, mels, num_frames)
        loss = causal_mel_loss + non_causal_mel_loss
        loss.backward()

        # update
        optim.minimize(loss)

        # logging
        tqdm.tqdm.write("[train] step: {}\tloss: {:.6f}\tcausal:{:.6f}\tnon_causal:{:.6f}".format(
            global_step, 
            loss.numpy()[0], 
            causal_mel_loss.numpy()[0], 
            non_causal_mel_loss.numpy()[0]))
        writer.add_scalar("loss/causal_mel_loss", causal_mel_loss.numpy()[0], global_step=global_step)
        writer.add_scalar("loss/non_causal_mel_loss", non_causal_mel_loss.numpy()[0], global_step=global_step)
        writer.add_scalar("loss/loss", loss.numpy()[0], global_step=global_step)
        
        if global_step % config["report_interval"] == 0:
            text_length = int(text_lengths.numpy()[0])
            num_frame = int(num_frames.numpy()[0])

            tag = "train_mel/ground-truth"
            img = cm.viridis(normalize(mels.numpy()[0, :num_frame].T))
            writer.add_image(tag, img, global_step=global_step, dataformats="HWC")

            tag = "train_mel/decoded"
            img = cm.viridis(normalize(decoded.numpy()[0, :num_frame].T))
            writer.add_image(tag, img, global_step=global_step, dataformats="HWC")

            tag = "train_mel/refined"
            img = cm.viridis(normalize(refined.numpy()[0, :num_frame].T))
            writer.add_image(tag, img, global_step=global_step, dataformats="HWC")

            vocoder = WaveflowVocoder()
            vocoder.model.eval()

            tag = "train_audio/ground-truth-waveflow"
            wav = vocoder(F.transpose(mels[0:1, :num_frame, :], (0, 2, 1)))
            writer.add_audio(tag, wav.numpy()[0], global_step=global_step, sample_rate=22050)

            tag = "train_audio/decoded-waveflow"
            wav = vocoder(F.transpose(decoded[0:1, :num_frame, :], (0, 2, 1)))
            writer.add_audio(tag, wav.numpy()[0], global_step=global_step, sample_rate=22050)

            tag = "train_audio/refined-waveflow"
            wav = vocoder(F.transpose(refined[0:1, :num_frame, :], (0, 2, 1)))
            writer.add_audio(tag, wav.numpy()[0], global_step=global_step, sample_rate=22050)
            
            attentions_np = attentions.numpy()
            attentions_np = attentions_np[:, 0, :num_frame // 4 , :text_length]
            for i, attention_layer in enumerate(np.rot90(attentions_np, axes=(1,2))):
                tag = "train_attention/layer_{}".format(i)
                img = cm.viridis(normalize(attention_layer))
                writer.add_image(tag, img, global_step=global_step, dataformats="HWC")

        if global_step % config["save_interval"] == 0:
            save_parameters(writer.logdir, global_step, model, optim)

        # global step +1
        global_step += 1

def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

if __name__ == "__main__":
    import argparse
    from ruamel import yaml

    parser = argparse.ArgumentParser(description="train a Deep Voice 3 model with LJSpeech")
    parser.add_argument("--config", type=str, required=True, help="config file")
    parser.add_argument("--input", type=str, required=True, help="data path of the original data")

    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        config = yaml.safe_load(f)
    
    dg.enable_dygraph(fluid.CUDAPlace(0))
    global global_step
    global_step = 1
    global writer
    writer = SummaryWriter()
    print("[Training] tensorboard log and checkpoints are save in {}".format(
        writer.logdir))
    train(args, config)