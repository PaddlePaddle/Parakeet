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

from operator import itemgetter
from typing import Any
from typing import Dict
from typing import List

import argparse
import jsonlines
import librosa
import logging
import numpy as np
import tqdm
from concurrent.futures import ThreadPoolExecutor
from parakeet.data.get_feats import LogMelFBank
from pathlib import Path
from praatio import tgio

from config import get_cfg_default


def process_sentence(config: Dict[str, Any],
                     fp: Path,
                     alignment_fp: Path,
                     output_dir: Path,
                     mel_extractor=None):
    utt_id = fp.stem

    # reading
    y, sr = librosa.load(str(fp), sr=config.sr)  # resampling may occur
    assert len(y.shape) == 1, f"{utt_id} is not a mono-channel audio."
    assert np.abs(
        y).max() <= 1.0, f"{utt_id} is seems to be different that 16 bit PCM."
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

    # extract mel feats
    logmel = mel_extractor.get_log_mel_fbank(y)

    # adjust time to make num_samples == num_frames * hop_length
    num_frames = logmel.shape[0]
    if y.size < num_frames * config.hop_length:
        y = np.pad(
            y, (0, num_frames * config.hop_length - y.size), mode="reflect")
    else:
        y = y[:num_frames * config.hop_length]
    num_sample = y.shape[0]

    mel_path = output_dir / (utt_id + "_feats.npy")
    wav_path = output_dir / (utt_id + "_wave.npy")
    np.save(wav_path, y)  # (num_samples, )
    np.save(mel_path, logmel)  # (num_frames, n_mels)
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
                      mel_extractor=None,
                      nprocs: int=1):
    if nprocs == 1:
        results = []
        for fp, alignment_fp in tqdm.tqdm(zip(fps, alignment_fps)):
            results.append(
                process_sentence(config, fp, alignment_fp, output_dir,
                                 mel_extractor))
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(fps)) as progress:
                for fp, alignment_fp in zip(fps, alignment_fps):
                    future = pool.submit(process_sentence, config, fp,
                                         alignment_fp, output_dir,
                                         mel_extractor)
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
        description="Preprocess audio and then extract features .")
    parser.add_argument(
        "--rootdir", default=None, type=str, help="directory to baker dataset.")
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
    num_train = 9800
    num_dev = 100

    train_wav_files = wav_files[:num_train]
    dev_wav_files = wav_files[num_train:num_train + num_dev]
    test_wav_files = wav_files[num_train + num_dev:]

    train_alignment_files = alignment_files[:num_train]
    dev_alignment_files = alignment_files[num_train:num_train + num_dev]
    test_alignment_files = alignment_files[num_train + num_dev:]

    train_dump_dir = dumpdir / "train" / "raw"
    train_dump_dir.mkdir(parents=True, exist_ok=True)
    dev_dump_dir = dumpdir / "dev" / "raw"
    dev_dump_dir.mkdir(parents=True, exist_ok=True)
    test_dump_dir = dumpdir / "test" / "raw"
    test_dump_dir.mkdir(parents=True, exist_ok=True)

    mel_extractor = LogMelFBank(
        sr=C.sr,
        n_fft=C.n_fft,
        hop_length=C.hop_length,
        win_length=C.win_length,
        window=C.window,
        n_mels=C.n_mels,
        fmin=C.fmin,
        fmax=C.fmax)

    # process for the 3 sections
    process_sentences(
        C,
        train_wav_files,
        train_alignment_files,
        train_dump_dir,
        mel_extractor=mel_extractor,
        nprocs=args.num_cpu)
    process_sentences(
        C,
        dev_wav_files,
        dev_alignment_files,
        dev_dump_dir,
        mel_extractor=mel_extractor,
        nprocs=args.num_cpu)
    process_sentences(
        C,
        test_wav_files,
        test_alignment_files,
        test_dump_dir,
        mel_extractor=mel_extractor,
        nprocs=args.num_cpu)


if __name__ == "__main__":
    main()
