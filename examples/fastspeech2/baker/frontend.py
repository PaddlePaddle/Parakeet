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
import numpy as np
import paddle
from pypinyin import lazy_pinyin, Style
import jieba


class Frontend():
    def __init__(self, vocab_path):

        self.voc_phones = {}
        with open(vocab_path, 'rt') as f:
            phn_id = [line.strip().split() for line in f.readlines()]
        for phn, id in phn_id:
            self.voc_phones[phn] = int(id)

    def segment(self, sentence):
        segments = re.split(r'[：，；。？！]', sentence)
        segments = [seg for seg in segments if len(seg)]
        return segments

    def g2p(self, sentence):
        segments = self.segment(sentence)
        phones = []

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
                if v:
                    phones.append(v)
            # add sp between sentence
            phones.append('sp')
        # replace last sp with <eos>
        phones[-1] = '<eos>'
        return phones

    def p2id(self, phonemes):
        # replace unk phone with sp
        phonemes = [
            phn if phn in self.voc_phones else "sp" for phn in phonemes
        ]
        phone_ids = [self.voc_phones[item] for item in phonemes]
        return np.array(phone_ids, np.int64)

    def text_analysis(self, sentence):
        phonemes = self.g2p(sentence)
        phone_ids = self.p2id(phonemes)
        phone_ids = paddle.to_tensor(phone_ids)
        return phone_ids
