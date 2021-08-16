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
from pathlib import Path

import numpy as np
import paddle
import pypinyin
from pypinyin import lazy_pinyin, Style
import jieba
import phkit
phkit.initialize()
from parakeet.frontend.vocab import Vocab

file_dir = Path(__file__).parent.resolve()
with open(file_dir / "phones.txt", 'rt') as f:
    phones = [line.strip() for line in f.readlines()]

with open(file_dir / "tones.txt", 'rt') as f:
    tones = [line.strip() for line in f.readlines()]
voc_phones = Vocab(phones, start_symbol=None, end_symbol=None)
voc_tones = Vocab(tones, start_symbol=None, end_symbol=None)


def segment(sentence):
    segments = re.split(r'[：，；。？！]', sentence)
    segments = [seg for seg in segments if len(seg)]
    return segments


def g2p(sentence):
    segments = segment(sentence)
    phones = []
    phones.append('sil')
    tones = []
    tones.append('0')

    for seg in segments:
        seg = jieba.lcut(seg)
        initials = lazy_pinyin(
            seg, neutral_tone_with_five=True, style=Style.INITIALS)
        finals = lazy_pinyin(
            seg, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
        for c, v in zip(initials, finals):
            # NOTE: post process for pypinyin outputs
            # we discriminate i, ii and iii
            if re.match(r'i\d', v):
                if c in ['z', 'c', 's']:
                    v = re.sub('i', 'ii', v)
                elif c in ['zh', 'ch', 'sh', 'r']:
                    v = re.sub('i', 'iii', v)
            if c:
                phones.append(c)
                tones.append('0')
            if v:
                phones.append(v[:-1])
                tones.append(v[-1])
        phones.append('sp')
        tones.append('0')
    phones[-1] = 'sil'
    tones[-1] = '0'
    return (phones, tones)


def p2id(voc, phonemes):
    phone_ids = [voc.lookup(item) for item in phonemes]
    return np.array(phone_ids, np.int64)


def t2id(voc, tones):
    tone_ids = [voc.lookup(item) for item in tones]
    return np.array(tone_ids, np.int64)


def text_analysis(sentence):
    phonemes, tones = g2p(sentence)
    print(sentence)
    print([p + t if t != '0' else p for p, t in zip(phonemes, tones)])
    phone_ids = p2id(voc_phones, phonemes)
    tone_ids = t2id(voc_tones, tones)
    phones = paddle.to_tensor(phone_ids)
    tones = paddle.to_tensor(tone_ids)
    return phones, tones
