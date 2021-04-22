from pathlib import Path
from multiprocessing import Pool
from functools import partial

import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from praatio import tgio


def get_valid_part(fpath):
    f = tgio.openTextgrid(fpath)

    start = 0
    phone_entry_list = f.tierDict['phones'].entryList
    first_entry = phone_entry_list[0]
    if first_entry.label == "sil":
        start = first_entry.end

    last_entry = phone_entry_list[-1]
    if last_entry.label == "sp":
        end = last_entry.start
    else:
        end = last_entry.end
    return start, end


def process_utterance(fpath, source_dir, target_dir, alignment_dir):
    rel_path = fpath.relative_to(source_dir)
    opath = target_dir / rel_path
    apath = (alignment_dir / rel_path).with_suffix(".TextGrid")
    opath.parent.mkdir(parents=True, exist_ok=True)

    start, end = get_valid_part(apath)
    wav, _ = librosa.load(fpath, sr=22050, offset=start, duration=end - start)
    normalized_wav = wav / np.max(wav) * 0.999
    sf.write(opath, normalized_wav, samplerate=22050, subtype='PCM_16')
    # print(f"{fpath} => {opath}")


def preprocess_aishell3(source_dir, target_dir, alignment_dir):
    source_dir = Path(source_dir).expanduser()
    target_dir = Path(target_dir).expanduser()
    alignment_dir = Path(alignment_dir).expanduser()

    wav_paths = list(source_dir.rglob("*.wav"))
    print(f"there are {len(wav_paths)} audio files in total")
    fx = partial(process_utterance,
                 source_dir=source_dir,
                 target_dir=target_dir,
                 alignment_dir=alignment_dir)
    with Pool(16) as p:
        list(
            tqdm(p.imap(fx, wav_paths), total=len(wav_paths),
                 unit="utterance"))


if __name__ == "__main__":
    preprocess_aishell3("~/datasets/aishell3/train/wav",
                        "~/datasets/aishell3/train/normalized_wav",
                        "~/datasets/aishell3/train/alignment")
