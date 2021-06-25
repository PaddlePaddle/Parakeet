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

import re
import argparse
from collections import OrderedDict

INITIALS = [
    'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'zh', 'ch', 'sh',
    'r', 'z', 'c', 's', 'j', 'q', 'x'
]

FINALS = [
    'a', 'ai', 'ao', 'an', 'ang', 'e', 'er', 'ei', 'en', 'eng', 'o', 'ou',
    'ong', 'ii', 'iii', 'i', 'ia', 'iao', 'ian', 'iang', 'ie', 'io', 'iou',
    'iong', 'in', 'ing', 'u', 'ua', 'uai', 'uan', 'uang', 'uei', 'uo', 'uen',
    'ueng', 'v', 've', 'van', 'vn'
]

SPECIALS = ['sil', 'sp']


def rule(C, V, R, T):

    # 不可拼的音节, ii 只能和 z, c, s 拼
    if V in ["ii"] and (C not in ['z', 'c', 's']):
        return
    # iii 只能和 zh, ch, sh, r 拼
    if V in ['iii'] and (C not in ['zh', 'ch', 'sh', 'r']):
        return

    # 齐齿呼或者撮口呼不能和 f, g, k, h, zh, ch, sh, r, z, c, s
    if (V not in ['ii', 'iii']) and V[0] in ['i', 'v'] and (
            C in ['f', 'g', 'k', 'h', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's']):
        return

    # 撮口呼只能和 j, q, x l, n 拼
    if V.startswith("v"):
        # v, ve 只能和 j ,q , x, n, l 拼
        if V in ['v', 've']:
            if C not in ['j', 'q', 'x', 'n', 'l', '']:
                return
        # 其他只能和 j, q, x 拼
        else:
            if C not in ['j', 'q', 'x', '']:
                return

    # j, q, x 只能和齐齿呼或者撮口呼拼
    if (C in ['j', 'q', 'x']) and not (
        (V not in ['ii', 'iii']) and V[0] in ['i', 'v']):
        return

    # b, p ,m, f 不能和合口呼拼，除了 u 之外
    # bm p, m, f 不能和撮口呼拼
    if (C in ['b', 'p', 'm', 'f']) and ((V[0] in ['u', 'v'] and V != "u") or
                                        V == 'ong'):
        return

    # ua, uai, uang 不能和 d, t, n, l, r, z, c, s 拼
    if V in ['ua', 'uai', 'uang'
             ] and C in ['d', 't', 'n', 'l', 'r', 'z', 'c', 's']:
        return

    # sh 和 ong 不能拼
    if V == 'ong' and C in ['sh']:
        return

    # o 和 gkh, zh ch sh r z c s 不能拼
    if V == "o" and C in [
            'd', 't', 'n', 'g', 'k', 'h', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's'
    ]:
        return

    # ueng 只是 weng 这个 ad-hoc 其他情况下都是 ong
    if V == 'ueng' and C != '':
        return

    # 非儿化的 er 只能单独存在
    if V == 'er' and C != '':
        return

    if C == '':
        if V in ["i", "in", "ing"]:
            C = 'y'
        elif V == 'u':
            C = 'w'
        elif V.startswith('i') and V not in ["ii", "iii"]:
            C = 'y'
            V = V[1:]
        elif V.startswith('u'):
            C = 'w'
            V = V[1:]
        elif V.startswith('v'):
            C = 'yu'
            V = V[1:]
    else:
        if C in ['j', 'q', 'x']:
            if V.startswith('v'):
                V = re.sub('v', 'u', V)
        if V == 'iou':
            V = 'iu'
        elif V == 'uei':
            V = 'ui'
        elif V == 'uen':
            V = 'un'
    result = C + V

    # Filter  er 不能再儿化
    if result.endswith('r') and R == 'r':
        return

    # ii and iii, change back to i
    result = re.sub(r'i+', 'i', result)

    result = result + R + T
    return result


def generate_lexicon(with_tone=False, with_r=False):
    # generate lexicon withou tone and erhua
    syllables = OrderedDict()

    for C in [''] + INITIALS:
        for V in FINALS:
            for R in [''] if not with_r else ['', 'r']:
                for T in [''] if not with_tone else ['1', '2', '3', '4', '5']:
                    result = rule(C, V, R, T)
                    if result:
                        syllables[result] = f'{C} {V}{R}{T}'
    return syllables


def generate_symbols(lexicon):
    symbols = set()
    for p in SPECIALS:
        symbols.add(p)
    for syllable, phonems in lexicon.items():
        phonemes = phonems.split()
        for p in phonemes:
            symbols.add(p)
    return sorted(list(symbols))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate lexicon for Chinese pinyin to phoneme for MFA")
    parser.add_argument("output", type=str, help="Path to save lexicon.")
    parser.add_argument(
        "--with-tone", action="store_true", help="whether to consider tone.")
    parser.add_argument(
        "--with-r", action="store_true", help="whether to consider erhua.")
    args = parser.parse_args()

    lexicon = generate_lexicon(args.with_tone, args.with_r)
    symbols = generate_symbols(lexicon)

    with open(args.output + ".lexicon", 'wt') as f:
        for k, v in lexicon.items():
            f.write(f"{k} {v}\n")

    with open(args.output + ".symbols", 'wt') as f:
        for s in symbols:
            f.write(s + "\n")

    print("Done!")
