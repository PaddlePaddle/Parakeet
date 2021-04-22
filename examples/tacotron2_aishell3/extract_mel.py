import argparse
import numpy as np
from pathlib import Path
from parakeet.audio import AudioProcessor
from parakeet.audio.spec_normalizer import NormalizerBase, LogMagnitude
import multiprocessing as mp
from functools import partial
import tqdm

from config import get_cfg_defaults


def extract_mel(fname: Path, input_dir: Path, output_dir: Path,
                p: AudioProcessor, n: NormalizerBase):
    relative_path = fname.relative_to(input_dir)
    out_path = (output_dir / relative_path).with_suffix(".npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wav = p.read_wav(fname)
    mel = p.mel_spectrogram(wav)
    mel = n.transform(mel)
    np.save(out_path, mel)


def extract_mel_multispeaker(config, input_dir, output_dir, extension=".wav"):
    input_dir = Path(input_dir).expanduser()
    fnames = list(input_dir.rglob("*.wav"))
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    p = AudioProcessor(config.sample_rate, config.n_fft, config.win_length,
                       config.hop_length, config.n_mels, config.fmin,
                       config.fmax)
    n = LogMagnitude(1e-5)

    func = partial(extract_mel,
                   input_dir=input_dir,
                   output_dir=output_dir,
                   p=p,
                   n=n)

    with mp.Pool(16) as pool:
        list(
            tqdm.tqdm(pool.imap(func, fnames),
                      total=len(fnames),
                      unit="utterance"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Extract mel spectrogram from processed wav in AiShell3 training dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        help="yaml config file to overwrite the default config")
    parser.add_argument("--input",
                        type=str,
                        default="~/datasets/aishell3/train/normalized_wav",
                        help="path of the processed wav folder")
    parser.add_argument("--output",
                        type=str,
                        default="~/datasets/aishell3/train/mel",
                        help="path of the folder to save mel spectrograms")
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        help=
        "options to overwrite --config file and the default config, passing in KEY VALUE pairs"
    )
    default_config = get_cfg_defaults()

    args = parser.parse_args()
    if args.config:
        default_config.merge_from_file(args.config)
    if args.opts:
        default_config.merge_from_list(args.opts)
    default_config.freeze()
    audio_config = default_config.data

    extract_mel_multispeaker(audio_config, args.input, args.output)
