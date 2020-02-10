import os
from tensorboardX import SummaryWriter
from collections import OrderedDict
import jsonargparse
from parse import add_config_options_to_parser
from pprint import pprint
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
from parakeet.g2p.en import text_to_sequence
from parakeet import audio
from network import FastSpeech

def load_checkpoint(step, model_path):
    model_dict, _ = fluid.dygraph.load_dygraph(os.path.join(model_path, step))
    new_state_dict = OrderedDict()
    for param in model_dict:
        if param.startswith('_layers.'):
            new_state_dict[param[8:]] = model_dict[param]
        else:
            new_state_dict[param] = model_dict[param]
    return new_state_dict

def synthesis(text_input, cfg):
    place = (fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace())

    # tensorboard
    if not os.path.exists(cfg.log_dir):
            os.mkdir(cfg.log_dir)
    path = os.path.join(cfg.log_dir,'synthesis')

    writer = SummaryWriter(path)

    with dg.guard(place):
        model = FastSpeech(cfg)
        model.set_dict(load_checkpoint(str(cfg.fastspeech_step), os.path.join(cfg.checkpoint_path, "fastspeech")))
        model.eval()

        text = np.asarray(text_to_sequence(text_input))
        text = fluid.layers.unsqueeze(dg.to_variable(text),[0])
        pos_text = np.arange(1, text.shape[1]+1)
        pos_text = fluid.layers.unsqueeze(dg.to_variable(pos_text),[0])

        mel_output, mel_output_postnet = model(text, pos_text, alpha=cfg.alpha)

        _ljspeech_processor = audio.AudioProcessor(
            sample_rate=cfg.audio.sr, 
            num_mels=cfg.audio.num_mels, 
            min_level_db=cfg.audio.min_level_db, 
            ref_level_db=cfg.audio.ref_level_db, 
            n_fft=cfg.audio.n_fft, 
            win_length= cfg.audio.win_length, 
            hop_length= cfg.audio.hop_length,
            power=cfg.audio.power,
            preemphasis=cfg.audio.preemphasis,
            signal_norm=True,
            symmetric_norm=False,
            max_norm=1.,
            mel_fmin=0,
            mel_fmax=None,
            clip_norm=True,
            griffin_lim_iters=60,
            do_trim_silence=False,
            sound_norm=False)

        mel_output_postnet = fluid.layers.transpose(fluid.layers.squeeze(mel_output_postnet,[0]), [1,0])
        wav = _ljspeech_processor.inv_melspectrogram(mel_output_postnet.numpy())
        writer.add_audio(text_input, wav, 0, cfg.audio.sr)
        print("Synthesis completed !!!")
    writer.close()

if __name__ == '__main__':
    parser = jsonargparse.ArgumentParser(description="Synthesis model", formatter_class='default_argparse')
    add_config_options_to_parser(parser)
    cfg = parser.parse_args('-c ./config/synthesis.yaml'.split())
    synthesis("Transformer model is so fast!", cfg)