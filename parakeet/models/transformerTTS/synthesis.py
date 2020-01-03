import os
from scipy.io.wavfile import write
from parakeet.g2p.en import text_to_sequence
import numpy as np
from network import Model, ModelPostNet
from tqdm import tqdm
from tensorboardX import SummaryWriter
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
from preprocess import _ljspeech_processor
from pathlib import Path
import jsonargparse
from parse import add_config_options_to_parser
from pprint import pprint

def load_checkpoint(step, model_path):
    model_dict, opti_dict = fluid.dygraph.load_dygraph(os.path.join(model_path, step))
    return model_dict

def synthesis(text_input, cfg):
    place = (fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace())

    # tensorboard
    if not os.path.exists(cfg.log_dir):
            os.mkdir(cfg.log_dir)
    path = os.path.join(cfg.log_dir,'synthesis')

    writer = SummaryWriter(path)

    with dg.guard(place):
        model = Model(cfg)
        model_postnet = ModelPostNet(cfg)

        model.set_dict(load_checkpoint(str(cfg.transformer_step), os.path.join(cfg.checkpoint_path, "transformer")))
        model_postnet.set_dict(load_checkpoint(str(cfg.postnet_step), os.path.join(cfg.checkpoint_path, "postnet")))

        # init input
        text = np.asarray(text_to_sequence(text_input))
        text = fluid.layers.unsqueeze(dg.to_variable(text),[0])
        mel_input = dg.to_variable(np.zeros([1,1,80])).astype(np.float32)
        pos_text = np.arange(1, text.shape[1]+1)
        pos_text = fluid.layers.unsqueeze(dg.to_variable(pos_text),[0])
        

        model.eval()
        model_postnet.eval()

        pbar = tqdm(range(cfg.max_len))

        for i in pbar:
            pos_mel = np.arange(1, mel_input.shape[1]+1)
            pos_mel = fluid.layers.unsqueeze(dg.to_variable(pos_mel),[0])
            mel_pred, postnet_pred, attn_probs, stop_preds, attn_enc, attn_dec = model(text, mel_input, pos_text, pos_mel)
            mel_input = fluid.layers.concat([mel_input, postnet_pred[:,-1:,:]], axis=1)
        mag_pred = model_postnet(postnet_pred)

        wav = _ljspeech_processor.inv_spectrogram(fluid.layers.transpose(fluid.layers.squeeze(mag_pred,[0]), [1,0]).numpy())
        writer.add_audio(text_input, wav, 0, cfg.audio.sr)
        if not os.path.exists(cfg.sample_path):
            os.mkdir(cfg.sample_path)
        write(os.path.join(cfg.sample_path,'test.wav'), cfg.audio.sr, wav)

if __name__ == '__main__':
    parser = jsonargparse.ArgumentParser(description="Synthesis model", formatter_class='default_argparse')
    add_config_options_to_parser(parser)
    cfg = parser.parse_args('-c ./config/synthesis.yaml'.split())
    synthesis("Transformer model is so fast!", cfg)