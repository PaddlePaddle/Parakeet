import os
import tqdm
import pickle
import argparse
import numpy as np
from pathlib import Path

from parakeet.datasets import LJSpeechMetaData
from parakeet.audio import AudioProcessor, LogMagnitude
from parakeet.frontend import English

from config import get_cfg_defaults

def create_dataset(config, source_path, target_path, verbose=False):
    # create output dir
    target_path = Path(target_path).expanduser()
    mel_path = target_path / "mel"
    os.makedirs(mel_path, exist_ok=True)

    meta_data = LJSpeechMetaData(source_path)
    frontend = English()
    processor = AudioProcessor(
        sample_rate=config.data.sample_rate,
        n_fft=config.data.n_fft,
        n_mels=config.data.d_mel,
        win_length=config.data.win_length, 
        hop_length=config.data.hop_length,
        f_max=config.data.f_max)
    normalizer = LogMagnitude()
    
    records = []
    for (fname, text, _) in tqdm.tqdm(meta_data):
        wav = processor.read_wav(fname)
        mel = processor.mel_spectrogram(wav)
        mel = normalizer.transform(mel)
        phonemes = frontend.phoneticize(text)
        ids = frontend.numericalize(phonemes)
        mel_name = os.path.splitext(os.path.basename(fname))[0]

        # save mel spectrogram
        records.append((mel_name, text, phonemes, ids))
        np.save(mel_path / mel_name, mel)
    if verbose:
        print("save mel spectrograms into {}".format(mel_path))
    
    # save meta data as pickle archive
    with open(target_path / "metadata.pkl", 'wb') as f:
        pickle.dump(records, f)
        if verbose:
            print("saved metadata into {}".format(target_path / "metadata.pkl"))

    # also save meta data into text format for inspection
    with open(target_path / "metadata.txt", 'wt') as f:
        for mel_name, text, phonemes, _ in records:
            phoneme_str = "|".join(phonemes)
            f.write("{}\t{}\t{}\n".format(mel_name, text, phoneme_str))
        if verbose:
            print("saved metadata into {}".format(target_path / "metadata.txt"))
    
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create dataset")
    parser.add_argument("--config", type=str, metavar="FILE", help="extra config to overwrite the default config")
    parser.add_argument("--input", type=str, help="path of the ljspeech dataset")
    parser.add_argument("--output", type=str, help="path to save output dataset")
    parser.add_argument("--opts", nargs=argparse.REMAINDER,
        help="options to overwrite --config file and the default config, passing in KEY VALUE pairs"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="print msg")
    
    config = get_cfg_defaults()
    args = parser.parse_args()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config.data)

    create_dataset(config, args.input, args.output, args.verbose)
