import os
import argparse
import ruamel.yamls
import numpy as np
import soundfile as sf

from paddle import fluid
import paddle.fluid.layers as F
import paddle.fluid.dygraph as dg
from tensorboardX import SummaryWriter

from parakeet.g2p import en
from parakeet.utils.layer_tools import summary
from parakeet.modules.weight_norm import WeightNormWrapper

from utils import make_model, eval_model, plot_alignment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthsize waveform with a checkpoint.")
    parser.add_argument("-c", "--config", type=str, help="experiment config.")
    parser.add_argument("checkpoint", type=str, help="checkpoint to load.")
    parser.add_argument("text", type=str, help="text file to synthesize")
    parser.add_argument("output_path", type=str, help="path to save results")

    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        config = ruamel.yaml.safe_load(f)

    if args.device == -1:
        place = fluid.CPUPlace()
    else:
        place = fluid.CUDAPlace(args.device)

    with dg.guard(place):
        # =========================model=========================
        transform_config = config["transform"]
        replace_pronounciation_prob = transform_config[
            "replace_pronunciation_prob"]
        sample_rate = transform_config["sample_rate"]
        preemphasis = transform_config["preemphasis"]
        n_fft = transform_config["n_fft"]
        n_mels = transform_config["n_mels"]

        model_config = config["model"]
        downsample_factor = model_config["downsample_factor"]
        r = model_config["outputs_per_step"]
        n_speakers = model_config["n_speakers"]
        speaker_dim = model_config["speaker_embed_dim"]
        speaker_embed_std = model_config["speaker_embedding_weight_std"]
        n_vocab = en.n_vocab
        embed_dim = model_config["text_embed_dim"]
        linear_dim = 1 + n_fft // 2
        use_decoder_states = model_config[
            "use_decoder_state_for_postnet_input"]
        filter_size = model_config["kernel_size"]
        encoder_channels = model_config["encoder_channels"]
        decoder_channels = model_config["decoder_channels"]
        converter_channels = model_config["converter_channels"]
        dropout = model_config["dropout"]
        padding_idx = model_config["padding_idx"]
        embedding_std = model_config["embedding_weight_std"]
        max_positions = model_config["max_positions"]
        freeze_embedding = model_config["freeze_embedding"]
        trainable_positional_encodings = model_config[
            "trainable_positional_encodings"]
        use_memory_mask = model_config["use_memory_mask"]
        query_position_rate = model_config["query_position_rate"]
        key_position_rate = model_config["key_position_rate"]
        window_behind = model_config["window_behind"]
        window_ahead = model_config["window_ahead"]
        key_projection = model_config["key_projection"]
        value_projection = model_config["value_projection"]
        dv3 = make_model(n_speakers, speaker_dim, speaker_embed_std, embed_dim,
                         padding_idx, embedding_std, max_positions, n_vocab,
                         freeze_embedding, filter_size, encoder_channels,
                         n_mels, decoder_channels, r,
                         trainable_positional_encodings, use_memory_mask,
                         query_position_rate, key_position_rate, window_behind,
                         window_ahead, key_projection, value_projection,
                         downsample_factor, linear_dim, use_decoder_states,
                         converter_channels, dropout)

        state, _ = dg.load_dygraph(args.checkpoint)
        dv3.set_dict(state)

        for layer in dv3.sublayers():
            if isinstance(layer, WeightNormWrapper):
                layer.remove_weight_norm()

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        transform_config = config["transform"]
        c = transform_config["replace_pronunciation_prob"]
        sample_rate = transform_config["sample_rate"]
        min_level_db = transform_config["min_level_db"]
        ref_level_db = transform_config["ref_level_db"]
        preemphasis = transform_config["preemphasis"]
        win_length = transform_config["win_length"]
        hop_length = transform_config["hop_length"]

        synthesis_config = config["synthesis"]
        power = synthesis_config["power"]
        n_iter = synthesis_config["n_iter"]

        with open(args.text, "rt", encoding="utf-8") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                text = line[:-1]
                dv3.eval()
                wav, attn = eval_model(dv3, text, replace_pronounciation_prob,
                                       min_level_db, ref_level_db, power,
                                       n_iter, win_length, hop_length,
                                       preemphasis)
                plot_alignment(
                    attn,
                    os.path.join(args.output_path, "test_{}.png".format(idx)))
                sf.write(
                    os.path.join(args.output_path, "test_{}.wav".format(idx)),
                    wav, sample_rate)
