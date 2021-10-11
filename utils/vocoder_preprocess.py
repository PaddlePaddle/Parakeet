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
import numpy as np
import tqdm
import yaml
from concurrent.futures import ThreadPoolExecutor
from parakeet.data.get_feats import LogMelFBank
from pathlib import Path
from yacs.config import CfgNode


# speaker|utt_id|phn dur phn dur ...
def get_phn_dur(file_name):
    '''
    read MFA duration.txt
    Parameters
    ----------
    file_name : str or Path
        path of gen_duration_from_textgrid.py's result
    Returns
    ----------
    Dict
        sentence: {'utt': ([char], [int])}
    '''
    f = open(file_name, 'r')
    sentence = {}
    speaker_set = set()
    for line in f:
        line_list = line.strip().split('|')
        utt = line_list[0]
        speaker = line_list[1]
        p_d = line_list[-1]
        speaker_set.add(speaker)
        phn_dur = p_d.split()
        phn = phn_dur[::2]
        dur = phn_dur[1::2]
        assert len(phn) == len(dur)
        sentence[utt] = (phn, [int(i) for i in dur], speaker)
    f.close()
    return sentence, speaker_set


def merge_silence(sentence):
    '''
    merge silences, set <eos>
    Parameters
    ----------
    sentence : Dict
        sentence: {'utt': (([char], [int]), str)}
    '''
    for utt in sentence:
        cur_phn, cur_dur, speaker = sentence[utt]
        new_phn = []
        new_dur = []

        # merge sp and sil
        for i, p in enumerate(cur_phn):
            if i > 0 and 'sil' == p and cur_phn[i - 1] in {"sil", "sp"}:
                new_dur[-1] += cur_dur[i]
                new_phn[-1] = 'sil'
            else:
                new_phn.append(p)
                new_dur.append(cur_dur[i])

        for i, (p, d) in enumerate(zip(new_phn, new_dur)):
            if p in {"sp"}:
                if d < 14:
                    new_phn[i] = 'sp'
                else:
                    new_phn[i] = 'spl'

        assert len(new_phn) == len(new_dur)
        sentence[utt] = [new_phn, new_dur, speaker]


def process_sentence(config: Dict[str, Any],
                     fp: Path,
                     sentences: Dict,
                     output_dir: Path,
                     mel_extractor=None,
                     cut_sil: bool=True):
    utt_id = fp.stem
    record = None
    if utt_id in sentences:
        # reading, resampling may occur
        y, _ = librosa.load(str(fp), sr=config.fs)
        if len(y.shape) != 1 or np.abs(y).max() > 1.0:
            return record
        assert len(y.shape) == 1, f"{utt_id} is not a mono-channel audio."
        assert np.abs(y).max(
        ) <= 1.0, f"{utt_id} is seems to be different that 16 bit PCM."
        phones = sentences[utt_id][0]
        durations = sentences[utt_id][1]
        speaker = sentences[utt_id][2]
        d_cumsum = np.pad(np.array(durations).cumsum(0), (1, 0), 'constant')
        # little imprecise than use *.TextGrid directly
        times = librosa.frames_to_time(
            d_cumsum, sr=config.fs, hop_length=config.n_shift)
        if cut_sil:
            start = 0
            end = d_cumsum[-1]
            if phones[0] == "sil" and len(durations) > 1:
                start = times[1]
                durations = durations[1:]
                phones = phones[1:]
            if phones[-1] == 'sil' and len(durations) > 1:
                end = times[-2]
                durations = durations[:-1]
                phones = phones[:-1]
            sentences[utt_id][0] = phones
            sentences[utt_id][1] = durations
            start, end = librosa.time_to_samples([start, end], sr=config.fs)
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
        if y.size < num_frames * config.n_shift:
            y = np.pad(
                y, (0, num_frames * config.n_shift - y.size), mode="reflect")
        else:
            y = y[:num_frames * config.n_shift]
        num_sample = y.shape[0]

        mel_path = output_dir / (utt_id + "_feats.npy")
        wav_path = output_dir / (utt_id + "_wave.npy")
        np.save(wav_path, y)  # (num_samples, )
        np.save(mel_path, logmel)  # (num_frames, n_mels)
        record = {
            "utt_id": utt_id,
            "num_samples": num_sample,
            "num_frames": num_frames,
            "feats": str(mel_path),
            "wave": str(wav_path),
        }
        return record


def process_sentences(config,
                      fps: List[Path],
                      sentences: Dict,
                      output_dir: Path,
                      mel_extractor=None,
                      nprocs: int=1,
                      cut_sil: bool=True):
    if nprocs == 1:
        results = []
        for fp in tqdm.tqdm(fps, total=len(fps)):
            record = process_sentence(config, fp, sentences, output_dir,
                                      mel_extractor, cut_sil)
            if record:
                results.append(record)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(fps)) as progress:
                for fp in fps:
                    future = pool.submit(process_sentence, config, fp,
                                         sentences, output_dir, mel_extractor,
                                         cut_sil)
                    future.add_done_callback(lambda p: progress.update())
                    futures.append(future)

                results = []
                for ft in futures:
                    record = ft.result()
                    if record:
                        results.append(record)

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
        "--dataset",
        default="baker",
        type=str,
        help="name of dataset, should in {baker, ljspeech} now")
    parser.add_argument(
        "--rootdir", default=None, type=str, help="directory to dataset.")
    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump feature files.")
    parser.add_argument("--config", type=str, help="vocoder config file.")
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)")
    parser.add_argument(
        "--num-cpu", type=int, default=1, help="number of process.")
    parser.add_argument(
        "--dur-file", default=None, type=str, help="path to durations.txt.")

    def str2bool(str):
        return True if str.lower() == 'true' else False

    parser.add_argument(
        "--cut-sil",
        type=str2bool,
        default=True,
        help="whether cut sil in the edge of audio")
    args = parser.parse_args()

    rootdir = Path(args.rootdir).expanduser()
    dumpdir = Path(args.dumpdir).expanduser()
    # use absolute path
    dumpdir = dumpdir.resolve()
    dumpdir.mkdir(parents=True, exist_ok=True)
    dur_file = Path(args.dur_file).expanduser()

    assert rootdir.is_dir()
    assert dur_file.is_file()

    with open(args.config, 'rt') as f:
        config = CfgNode(yaml.safe_load(f))

    if args.verbose > 1:
        print(vars(args))
        print(config)

    sentences, speaker_set = get_phn_dur(dur_file)
    merge_silence(sentences)

    # split data into 3 sections
    if args.dataset == "baker":
        wav_files = sorted(list((rootdir / "Wave").rglob("*.wav")))
        num_train = 9800
        num_dev = 100

    elif args.dataset == "ljspeech":
        wav_files = sorted(list((rootdir / "wavs").rglob("*.wav")))
        # split data into 3 sections
        num_train = 12900
        num_dev = 100

    train_wav_files = wav_files[:num_train]
    dev_wav_files = wav_files[num_train:num_train + num_dev]
    test_wav_files = wav_files[num_train + num_dev:]

    train_dump_dir = dumpdir / "train" / "raw"
    train_dump_dir.mkdir(parents=True, exist_ok=True)
    dev_dump_dir = dumpdir / "dev" / "raw"
    dev_dump_dir.mkdir(parents=True, exist_ok=True)
    test_dump_dir = dumpdir / "test" / "raw"
    test_dump_dir.mkdir(parents=True, exist_ok=True)

    mel_extractor = LogMelFBank(
        sr=config.fs,
        n_fft=config.n_fft,
        hop_length=config.n_shift,
        win_length=config.win_length,
        window=config.window,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax)

    # process for the 3 sections
    if train_wav_files:
        process_sentences(
            config,
            train_wav_files,
            sentences,
            train_dump_dir,
            mel_extractor=mel_extractor,
            nprocs=args.num_cpu,
            cut_sil=args.cut_sil)
    if dev_wav_files:
        process_sentences(
            config,
            dev_wav_files,
            sentences,
            dev_dump_dir,
            mel_extractor=mel_extractor,
            nprocs=args.num_cpu,
            cut_sil=args.cut_sil)
    if test_wav_files:
        process_sentences(
            config,
            test_wav_files,
            sentences,
            test_dump_dir,
            mel_extractor=mel_extractor,
            nprocs=args.num_cpu,
            cut_sil=args.cut_sil)


if __name__ == "__main__":
    main()
