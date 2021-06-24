# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Dict, Any
import soundfile as sf
import librosa
import numpy as np
import argparse
import yaml
import json
import jsonlines
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import tqdm
from operator import itemgetter
from praatio import tgio
import logging

from config import get_cfg_default


def logmelfilterbank(audio,
                     sr,
                     n_fft=1024,
                     hop_length=256,
                     win_length=None,
                     window="hann",
                     n_mels=80,
                     fmin=None,
                     fmax=None,
                     eps=1e-10):
    """Compute log-Mel filterbank feature.

    Parameters
    ----------
    audio : ndarray
        Audio signal (T,).
    sr : int
        Sampling rate.
    n_fft : int
        FFT size. (Default value = 1024)
    hop_length : int
        Hop size. (Default value = 256)
    win_length : int
        Window length. If set to None, it will be the same as fft_size. (Default value = None)
    window : str
        Window function type. (Default value = "hann")
    n_mels : int
        Number of mel basis. (Default value = 80)
    fmin : int
        Minimum frequency in mel basis calculation. (Default value = None)
    fmax : int
        Maximum frequency in mel basis calculation. (Default value = None)
    eps : float
        Epsilon value to avoid inf in log calculation. (Default value = 1e-10)

    Returns
    -------
    np.ndarray
        Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        pad_mode="reflect")
    spc = np.abs(x_stft)  # (#bins, #frames,)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sr / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)

    return np.log10(np.maximum(eps, np.dot(mel_basis, spc)))


def process_sentence(config: Dict[str, Any],
                     fp: Path,
                     alignment_fp: Path,
                     output_dir: Path):
    utt_id = fp.stem

    # reading
    y, sr = librosa.load(fp, sr=config.sr)  # resampling may occur
    assert len(y.shape) == 1, f"{utt_id} is not a mono-channel audio."
    assert np.abs(y).max(
    ) <= 1.0, f"{utt_id} is seems to be different that 16 bit PCM."
    duration = librosa.get_duration(y, sr=sr)

    # trim according to the alignment file
    alignment = tgio.openTextgrid(alignment_fp)
    intervals = alignment.tierDict[alignment.tierNameList[0]].entryList
    first, last = intervals[0], intervals[-1]
    start = 0
    end = last.end
    if first.label == "sil" and first.end < duration:
        start = first.end
    else:
        logging.warning(
            f" There is something wrong with the fisrt interval {first} in utterance: {utt_id}"
        )
    if last.label == "sil" and last.start < duration:
        end = last.start
    else:
        end = duration
        logging.warning(
            f" There is something wrong with the last interval {last} in utterance: {utt_id}"
        )
    # silence trimmed
    start, end = librosa.time_to_samples([first.end, last.start], sr=sr)
    y = y[start:end]

    # energy based silence trimming
    if config.trim_silence:
        y, _ = librosa.effects.trim(
            y,
            top_db=config.top_db,
            frame_length=config.trim_frame_length,
            hop_length=config.trim_hop_length)

    logmel = logmelfilterbank(
        y,
        sr=sr,
        n_fft=config.n_fft,
        window=config.window,
        win_length=config.win_length,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax)

    # adjust time to make num_samples == num_frames * hop_length
    num_frames = logmel.shape[1]
    y = np.pad(y, (0, config.n_fft), mode="reflect")
    y = y[:num_frames * config.hop_length]
    num_sample = y.shape[0]

    mel_path = output_dir / (utt_id + "_feats.npy")
    wav_path = output_dir / (utt_id + "_wave.npy")
    np.save(wav_path, y)  # (num_samples, )
    np.save(mel_path, logmel.T)  # (num_frames, n_mels)
    record = {
        "utt_id": utt_id,
        "num_samples": num_sample,
        "num_frames": num_frames,
        "feats": str(mel_path.resolve()),
        "wave": str(wav_path.resolve()),
    }
    return record


def process_sentences(config,
                      fps: List[Path],
                      alignment_fps: List[Path],
                      output_dir: Path,
                      nprocs: int=1):
    if nprocs == 1:
        results = []
        for fp, alignment_fp in tqdm.tqdm(zip(fps, alignment_fps)):
            results.append(
                process_sentence(config, fp, alignment_fp, output_dir))
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(fps)) as progress:
                for fp, alignment_fp in zip(fps, alignment_fps):
                    future = pool.submit(process_sentence, config, fp,
                                         alignment_fp, output_dir)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    results.append(ft.result())

    results.sort(key=itemgetter("utt_id"))
    with jsonlines.open(output_dir / "metadata.jsonl", 'w') as writer:
        for item in results:
            writer.write(item)
    print("Done")


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features (See detail in parallel_wavegan/bin/preprocess.py)."
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        help="directory including wav files. you need to specify either scp or rootdir."
    )
    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump feature files.")
    parser.add_argument(
        "--config", type=str, help="yaml format configuration file.")
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)")
    parser.add_argument(
        "--num_cpu", type=int, default=1, help="number of process.")
    args = parser.parse_args()

    C = get_cfg_default()
    if args.config:
        C.merge_from_file(args.config)
        C.freeze()

    if args.verbose > 1:
        print(vars(args))
        print(C)

    root_dir = Path(args.rootdir).expanduser()
    dumpdir = Path(args.dumpdir).expanduser()
    dumpdir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(list((root_dir / "Wave").rglob("*.wav")))
    alignment_files = sorted(
        list((root_dir / "PhoneLabeling").rglob("*.interval")))

    # split data into 3 sections
    train_wav_files = wav_files[:9800]
    dev_wav_files = wav_files[9800:9900]
    test_wav_files = wav_files[9900:]

    train_alignment_files = alignment_files[:9800]
    dev_alignment_files = alignment_files[9800:9900]
    test_alignment_files = alignment_files[9900:]

    train_dump_dir = dumpdir / "train" / "raw"
    train_dump_dir.mkdir(parents=True, exist_ok=True)
    dev_dump_dir = dumpdir / "dev" / "raw"
    dev_dump_dir.mkdir(parents=True, exist_ok=True)
    test_dump_dir = dumpdir / "test" / "raw"
    test_dump_dir.mkdir(parents=True, exist_ok=True)

    # process for the 3 sections
    process_sentences(
        C,
        train_wav_files,
        train_alignment_files,
        train_dump_dir,
        nprocs=args.num_cpu)
    process_sentences(
        C,
        dev_wav_files,
        dev_alignment_files,
        dev_dump_dir,
        nprocs=args.num_cpu)
    process_sentences(
        C,
        test_wav_files,
        test_alignment_files,
        test_dump_dir,
        nprocs=args.num_cpu)


if __name__ == "__main__":
    main()
