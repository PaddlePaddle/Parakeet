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
import re
from collections import defaultdict
from pathlib import Path

from parakeet.frontend.cn_frontend import Frontend as cnFrontend
from parakeet.utils.error_rate import wer
from praatio import tgio


def text_cleaner(raw_text):
    text = re.sub('#[1-4]|“|”|（|）', '', raw_text)
    text = text.replace("…。", "。")
    text = re.sub('：|；|——|……|、|…|—', '，', text)
    return text


def get_baker_data(root_dir):
    alignment_files = sorted(
        list((root_dir / "PhoneLabeling").rglob("*.interval")))
    text_file = root_dir / "ProsodyLabeling/000001-010000.txt"
    text_file = Path(text_file).expanduser()
    data_dict = defaultdict(dict)
    # filter out several files that have errors in annotation
    exclude = {'000611', '000662', '002365', '005107'}
    alignment_files = [f for f in alignment_files if f.stem not in exclude]
    # biaobei 前后有 sil ，中间没有 sp
    data_dict = defaultdict(dict)
    for alignment_fp in alignment_files:
        alignment = tgio.openTextgrid(alignment_fp)
        # only with baker's annotation
        utt_id = alignment.tierNameList[0].split(".")[0]
        intervals = alignment.tierDict[alignment.tierNameList[0]].entryList
        phones = []
        for interval in intervals:
            label = interval.label
            # Baker has sp1 rather than sp
            label = label.replace("sp1", "sp")
            phones.append(label)
        data_dict[utt_id]["phones"] = phones
    for line in open(text_file, "r"):
        if line.startswith("0"):
            utt_id, raw_text = line.strip().split()
            text = text_cleaner(raw_text)
            if utt_id in data_dict:
                data_dict[utt_id]['text'] = text
        else:
            pinyin = line.strip().split()
            if utt_id in data_dict:
                data_dict[utt_id]['pinyin'] = pinyin
    return data_dict


def get_g2p_phones(data_dict, frontend):
    for utt_id in data_dict:
        g2p_phones = frontend.get_phonemes(data_dict[utt_id]['text'])
        data_dict[utt_id]["g2p_phones"] = g2p_phones
    return data_dict


def get_avg_wer(data_dict):
    wer_list = []
    for utt_id in data_dict:
        g2p_phones = data_dict[utt_id]['g2p_phones']
        # delete silence tokens in predicted phones
        g2p_phones = [phn for phn in g2p_phones if phn not in {"sp", "sil"}]
        gt_phones = data_dict[utt_id]['phones']
        # delete silence tokens in baker phones
        gt_phones = [phn for phn in gt_phones if phn not in {"sp", "sil"}]
        gt_phones = " ".join(gt_phones)
        g2p_phones = " ".join(g2p_phones)
        single_wer = wer(gt_phones, g2p_phones)
        wer_list.append(single_wer)
    return sum(wer_list) / len(wer_list)


def main():
    parser = argparse.ArgumentParser(description="g2p example.")
    parser.add_argument(
        "--root-dir",
        default=None,
        type=str,
        help="directory to baker dataset.")

    args = parser.parse_args()
    root_dir = Path(args.root_dir).expanduser()
    assert root_dir.is_dir()
    frontend = cnFrontend()
    data_dict = get_baker_data(root_dir)
    data_dict = get_g2p_phones(data_dict, frontend)
    avg_wer = get_avg_wer(data_dict)
    print("The avg WER of g2p is:", avg_wer)


if __name__ == "__main__":
    main()
