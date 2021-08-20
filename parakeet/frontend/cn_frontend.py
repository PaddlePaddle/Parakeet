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
from typing import List

import jieba.posseg as psg
from g2pM import G2pM
from pypinyin import lazy_pinyin
from pypinyin import Style

from parakeet.frontend.cn_normalization.text_normlization import TextNormalizer
from parakeet.frontend.generate_lexicon import generate_lexicon
from parakeet.frontend.tone_sandhi import ToneSandhi


class Frontend():
    def __init__(self, g2p_model="pypinyin"):
        self.tone_modifier = ToneSandhi()
        self.text_normalizer = TextNormalizer()
        self.punc = "：，；。？！“”‘’':,;.?!"
        # g2p_model can be pypinyin and g2pM
        self.g2p_model = g2p_model
        if self.g2p_model == "g2pM":
            self.g2pM_model = G2pM()
            self.pinyin2phone = generate_lexicon(
                with_tone=True, with_erhua=False)
        self.must_erhua = {"小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿"}
        self.not_erhua = {
            "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿", "妻儿",
            "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿", "脑瘫儿",
            "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿", "可儿", "侄儿",
            "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿", "鸟儿", "猪儿", "猫儿",
            "狗儿"
        }

    def _get_initials_finals(self, word: str) -> List[List[str]]:
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
                pinyin = pinyin.replace("u:", "v")
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
    def _g2p(self,
             sentences: List[str],
             merge_sentences: bool=True,
             with_erhua: bool=True) -> List[List[str]]:
        segments = sentences
        phones_list = []
        for seg in segments:
            phones = []
            seg_cut = psg.lcut(seg)
            initials = []
            finals = []
            seg_cut = self.tone_modifier.pre_merge_for_modify(seg_cut)
            for word, pos in seg_cut:
                if pos == 'eng':
                    continue
                sub_initials, sub_finals = self._get_initials_finals(word)

                sub_finals = self.tone_modifier.modified_tone(word, pos,
                                                              sub_finals)
                if with_erhua:
                    sub_initials, sub_finals = self._merge_erhua(
                        sub_initials, sub_finals, word, pos)
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
            merge_list = sum(phones_list, [])
            phones_list = []
            phones_list.append(merge_list)
        return phones_list

    def _merge_erhua(self,
                     initials: List[str],
                     finals: List[str],
                     word: str,
                     pos: str) -> List[List[str]]:
        if word not in self.must_erhua and (word in self.not_erhua or
                                            pos in {"a", "j", "nr"}):
            return initials, finals
        new_initials = []
        new_finals = []
        assert len(finals) == len(word)
        for i, phn in enumerate(finals):
            if i == len(finals) - 1 and word[i] == "儿" and phn in {
                    "er2", "er5"
            } and word[-2:] not in self.not_erhua and new_finals:
                new_finals[-1] = new_finals[-1][:-1] + "r" + new_finals[-1][-1]
            else:
                new_finals.append(phn)
                new_initials.append(initials[i])
        return new_initials, new_finals

    def get_phonemes(self,
                     sentence: str,
                     merge_sentences: bool=True,
                     with_erhua: bool=True) -> List[List[str]]:
        sentences = self.text_normalizer.normalize(sentence)
        phonemes = self._g2p(
            sentences, merge_sentences=merge_sentences, with_erhua=with_erhua)
        return phonemes
