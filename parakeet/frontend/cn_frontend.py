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


import jieba.posseg as psg
import numpy as np
import paddle
import re
from g2pM import G2pM
from parakeet.frontend.tone_sandhi import ToneSandhi
from parakeet.frontend.cn_normalization.text_normlization import TextNormalizer
from pypinyin import lazy_pinyin, Style

from parakeet.frontend.generate_lexicon import generate_lexicon


class Frontend():
    def __init__(self, g2p_model="pypinyin"):
        self.tone_modifier = ToneSandhi()
        self.text_normalizer = TextNormalizer()
        self.punc = "：，；。？！“”‘’':,;.?!"
        # g2p_model can be pypinyin and g2pM
        self.g2p_model = g2p_model
        if self.g2p_model == "g2pM":
            self.g2pM_model = G2pM()
            self.pinyin2phone = generate_lexicon(with_tone=True, with_erhua=False)

    def _get_initials_finals(self, word):
        initials = []
        finals = []
        if self.g2p_model == "pypinyin":
            orig_initials = lazy_pinyin(
                word, neutral_tone_with_five=True, style=Style.INITIALS)
            orig_finals = lazy_pinyin(
                word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            for c, v in zip(orig_initials, orig_finals):
                if re.match(r'i\d', v):
                    if c in ['z', 'c', 's']:
                        v = re.sub('i', 'ii', v)
                    elif c in ['zh', 'ch', 'sh', 'r']:
                        v = re.sub('i', 'iii', v)
                initials.append(c)
                finals.append(v)
        elif self.g2p_model == "g2pM":
            pinyins = self.g2pM_model(word, tone=True, char_split=False)
            for pinyin in pinyins:
                pinyin = pinyin.replace("u:","v")
                if pinyin in self.pinyin2phone:
                    initial_final_list = self.pinyin2phone[pinyin].split(" ")
                    if len(initial_final_list) == 2:
                        initials.append(initial_final_list[0])
                        finals.append(initial_final_list[1])
                    elif len(initial_final_list) == 1:
                        initials.append('')
                        finals.append(initial_final_list[1])
                else:
                    # If it's not pinyin (possibly punctuation) or no conversion is required
                    initials.append(pinyin)
                    finals.append(pinyin)
        return initials, finals

    # if merge_sentences, merge all sentences into one phone sequence
    def _g2p(self, sentences, merge_sentences=True):
        segments = sentences
        phones_list = []
        for seg in segments:
            phones = []
            seg = psg.lcut(seg)
            initials = []
            finals = []
            seg = self.tone_modifier.pre_merge_for_modify(seg)
            for word, pos in seg:
                if pos == 'eng':
                    continue
                sub_initials, sub_finals = self._get_initials_finals(word)
                sub_finals = self.tone_modifier.modified_tone(word, pos, sub_finals)
                initials.append(sub_initials)
                finals.append(sub_finals)
                # assert len(sub_initials) == len(sub_finals) == len(word)
            initials = sum(initials, [])
            finals = sum(finals, [])
            for c, v in zip(initials, finals):
                # NOTE: post process for pypinyin outputs
                # we discriminate i, ii and iii
                if c and c not in self.punc:
                    phones.append(c)
                if v and v not in self.punc:
                    phones.append(v)
            # add sp between sentence (replace the last punc with sp)
            if initials[-1] in self.punc:
                phones.append('sp')
            phones_list.append(phones)
        if merge_sentences:
            phones_list = sum(phones_list, [])
        return phones_list

    def get_phonemes(self, sentence):
        sentences = self.text_normalizer.normalize(sentence)
        phonemes = self._g2p(sentences)
        return phonemes
