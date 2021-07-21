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
import os
from pathlib import Path

import librosa
import numpy as np
from praatio import tgio

from config import get_cfg_default


def readtg(config, tg_path):
    alignment = tgio.openTextgrid(tg_path, readRaw=True)
    phones = []
    ends = []
    for interval in alignment.tierDict["phones"].entryList:
        phone = interval.label
        phones.append(phone)
        ends.append(interval.end)
    frame_pos = librosa.time_to_frames(
        ends, sr=config.fs, hop_length=config.n_shift)
    durations = np.diff(frame_pos, prepend=0)
    assert len(durations) == len(phones)
    results = ""
    for (p, d) in zip(phones, durations):
        p = "sil" if p == "" else p
        results += p + " " + str(d) + " "
    return results.strip()


# assume that the directory structure of inputdir is inputdir/speaker/*.TextGrid
# in MFA1.x, there are blank labels("") in the end, we replace it with "sil"
def gen_duration_from_textgrid(config, inputdir, output):
    durations_dict = {}

    for speaker in os.listdir(inputdir):
        subdir = inputdir / speaker
        for file in os.listdir(subdir):
            if file.endswith(".TextGrid"):
                tg_path = subdir / file
                name = file.split(".")[0]
                durations_dict[name] = readtg(config, tg_path)
    with open(output, "w") as wf:
        for name in sorted(durations_dict.keys()):
            wf.write(name + "|" + durations_dict[name] + "\n")


def main():
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")
    parser.add_argument(
        "--inputdir",
        default=None,
        type=str,
        help="directory to alignment files.")
    parser.add_argument(
        "--output", type=str, required=True, help="output duration file name")
    parser.add_argument(
        "--config", type=str, help="yaml format configuration file.")

    args = parser.parse_args()
    C = get_cfg_default()
    if args.config:
        C.merge_from_file(args.config)
        C.freeze()

    inputdir = Path(args.inputdir).expanduser()
    output = Path(args.output).expanduser()
    gen_duration_from_textgrid(C, inputdir, output)


if __name__ == "__main__":
    main()
