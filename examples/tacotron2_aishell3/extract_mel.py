import numpy as np
from pathlib import Path
from parakeet.audio import AudioProcessor
from parakeet.audio.spec_normalizer import LogMagnitude
import multiprocessing as mp
from functools import partial
import tqdm
from yacs.config import CfgNode

def extract_mel(fname:Path, input_dir:Path, output_dir:Path, p, n):
    relative_path = fname.relative_to(input_dir)
    out_path = (output_dir / relative_path).with_suffix(".npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # TODO: maybe we need to rescale the audio
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
    audio_config = {
        "sample_rate": 22050,
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 256,
        "n_mels": 80,
        "fmin": 0,
        "fmax": 8000}
    audio_config = CfgNode(audio_config)
    extract_mel_multispeaker(audio_config, "~/datasets/aishell3/train/normalized_wav", "~/datasets/aishell3/train/mel")
    
