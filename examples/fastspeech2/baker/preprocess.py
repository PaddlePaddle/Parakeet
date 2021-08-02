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

import argparse
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
from pathlib import Path
from typing import List, Dict, Any

import jsonlines
import librosa
import numpy as np
import tqdm

from config import get_cfg_default
from get_feats import LogMelFBank, Energy, Pitch


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
    for line in f:
        utt = line.strip().split('|')[0]
        p_d = line.strip().split('|')[-1]
        phn_dur = p_d.split()
        phn = phn_dur[::2]
        dur = phn_dur[1::2]
        assert len(phn) == len(dur)
        sentence[utt] = (phn, [int(i) for i in dur])
    f.close()
    return sentence


def deal_silence(sentence):
    '''
    merge silences, set <eos>
    Parameters
    ----------
        sentence : Dict
            sentence: {'utt': ([char], [int])}
    '''
    for utt in sentence:
        cur_phn, cur_dur = sentence[utt]
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
        sentence[utt] = [new_phn, new_dur]


def get_input_token(sentence, output_path):
    '''
    get phone set from training data and save it
    Parameters
    ----------
        sentence : Dict
            sentence: {'utt': ([char], [int])}
        output_path : str or path
            path to save phone_id_map
    '''
    phn_emb = set()
    for utt in sentence:
        for phn in sentence[utt][0]:
            if phn != "<eos>":
                phn_emb.add(phn)
    phn_emb = list(phn_emb)
    phn_emb.sort()
    phn_emb = ["<pad>", "<unk>"] + phn_emb
    phn_emb += ["，", "。", "？", "！", "<eos>"]

    f = open(output_path, 'w')
    for i, phn in enumerate(phn_emb):
        f.write(phn + ' ' + str(i) + '\n')
    f.close()


def compare_duration_and_mel_length(sentences, utt, mel):
    '''
    check duration error, correct sentences[utt] if possible, else pop sentences[utt]
    Parameters
    ----------
        sentences : Dict
            sentences[utt] = [phones_list ,durations_list]
        utt : str
            utt_id
        mel : np.ndarry
            features (num_frames, n_mels)
    '''

    if utt in sentences:
        len_diff = mel.shape[0] - sum(sentences[utt][1])
        if len_diff != 0:
            if len_diff > 0:
                sentences[utt][1][-1] += len_diff
            elif sentences[utt][1][-1] + len_diff > 0:
                sentences[utt][1][-1] += len_diff
            elif sentences[utt][1][0] + len_diff > 0:
                sentences[utt][1][0] += len_diff
            else:
                print("the len_diff is unable to correct:", len_diff)
                sentences.pop(utt)


def process_sentence(
        config: Dict[str, Any],
        fp: Path,
        sentences: Dict,
        output_dir: Path,
        mel_extractor=None,
        pitch_extractor=None,
        energy_extractor=None,
        cut_sil: bool = True):
    utt_id = fp.stem
    record = None
    if utt_id in sentences:
        # reading, resampling may occur
        wav, _ = librosa.load(str(fp), sr=config.fs)
        assert len(wav.shape) == 1, f"{utt_id} is not a mono-channel audio."
        assert np.abs(wav).max(
        ) <= 1.0, f"{utt_id} is seems to be different that 16 bit PCM."
        phones = sentences[utt_id][0]
        durations = sentences[utt_id][1]
        d_cumsum = np.pad(np.array(durations).cumsum(0), (1, 0), 'constant')
        # little imprecise than use *.TextGrid directly
        times = librosa.frames_to_time(d_cumsum, sr=config.fs, hop_length=config.n_shift)
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
            wav = wav[start:end]
        # extract mel feats
        logmel = mel_extractor.get_log_mel_fbank(wav)
        # change duration according to mel_length
        compare_duration_and_mel_length(sentences, utt_id, logmel)
        phones = sentences[utt_id][0]
        durations = sentences[utt_id][1]
        num_frames = logmel.shape[0]
        assert sum(durations) == num_frames
        mel_dir = output_dir / "data_speech"
        mel_dir.mkdir(parents=True, exist_ok=True)
        mel_path = mel_dir / (utt_id + "_speech.npy")
        np.save(mel_path, logmel)
        # extract pitch and energy
        f0 = pitch_extractor.get_pitch(wav, duration=np.array(durations))
        assert f0.shape[0] == len(durations)
        f0_dir = output_dir / "data_pitch"
        f0_dir.mkdir(parents=True, exist_ok=True)
        f0_path = f0_dir / (utt_id + "_pitch.npy")
        np.save(f0_path, f0)
        energy = energy_extractor.get_energy(wav, duration=np.array(durations))
        assert energy.shape[0] == len(durations)
        energy_dir = output_dir / "data_energy"
        energy_dir.mkdir(parents=True, exist_ok=True)
        energy_path = energy_dir / (utt_id + "_energy.npy")
        np.save(energy_path, energy)
        record = {
            "utt_id": utt_id,
            "phones": phones,
            "text_lengths": len(phones),
            "speech_lengths": num_frames,
            "durations": durations,
            # use absolute path
            "speech": str(mel_path.resolve()),
            "pitch": str(f0_path.resolve()),
            "energy": str(energy_path.resolve())
        }
    return record


def process_sentences(config,
                      fps: List[Path],
                      sentences: Dict,
                      output_dir: Path,
                      mel_extractor=None,
                      pitch_extractor=None,
                      energy_extractor=None,
                      nprocs: int = 1,
                      cut_sil: bool = True):
    if nprocs == 1:
        results = []
        for fp in tqdm.tqdm(fps, total=len(fps)):
            record = process_sentence(config, fp, sentences, output_dir,
                                      mel_extractor, pitch_extractor,
                                      energy_extractor, cut_sil)
            if record:
                results.append(record)
    else:
        with ThreadPoolExecutor(nprocs) as pool:
            futures = []
            with tqdm.tqdm(total=len(fps)) as progress:
                for fp in fps:
                    future = pool.submit(process_sentence, config, fp,
                                         sentences, output_dir, mel_extractor,
                                         pitch_extractor, energy_extractor, cut_sil)
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
        description="Preprocess audio and then extract features.")
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        help="directory to baker dataset.")
    parser.add_argument(
        "--dur-path",
        default=None,
        type=str,
        help="path to baker durations.txt.")
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
        "--num-cpu", type=int, default=1, help="number of process.")
    def str2bool(str):
        return True if str.lower() == 'true' else False
    parser.add_argument(
        "--cut-sil", type=str2bool, default=True, help="whether cut sil in the edge of audio")
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

    sentences = get_phn_dur(args.dur_path)
    deal_silence(sentences)
    phone_id_map_path = dumpdir / "phone_id_map.txt"
    get_input_token(sentences, phone_id_map_path)
    wav_files = sorted(list((root_dir / "Wave").rglob("*.wav")))

    # split data into 3 sections
    num_train = 9800
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

    # Extractor
    mel_extractor = LogMelFBank(C)
    pitch_extractor = Pitch(C)
    energy_extractor = Energy(C)

    # process for the 3 sections
    
    process_sentences(
        C,
        train_wav_files,
        sentences,
        train_dump_dir,
        mel_extractor,
        pitch_extractor,
        energy_extractor,
        nprocs=args.num_cpu,
        cut_sil=args.cut_sil)
    process_sentences(
        C,
        dev_wav_files,
        sentences,
        dev_dump_dir,
        mel_extractor,
        pitch_extractor,
        energy_extractor,
        cut_sil=args.cut_sil)
    
    process_sentences(
        C,
        test_wav_files,
        sentences,
        test_dump_dir,
        mel_extractor,
        pitch_extractor,
        energy_extractor,
        nprocs=args.num_cpu,
        cut_sil=args.cut_sil)


if __name__ == "__main__":
    main()
