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

import opencc
import re
from typing import List

from .chronology import RE_TIME, RE_DATE, RE_DATE2
from .chronology import replace_time, replace_date, replace_date2
from .constants import F2H_ASCII_LETTERS, F2H_DIGITS, F2H_SPACE
from .num import RE_NUMBER, RE_FRAC, RE_PERCENTAGE, RE_RANGE, RE_INTEGER, RE_DEFAULT_NUM
from .num import replace_number, replace_frac, replace_percentage, replace_range, replace_default_num
from .phone import RE_MOBILE_PHONE, RE_TELEPHONE, replace_phone
from .quantifier import RE_TEMPERATURE
from .quantifier import replace_temperature


class Normalizer():
    def __init__(self):
        self.SENTENCE_SPLITOR = re.compile(r'([：，；。？！,;?!][”’]?)')
        self._t2s_converter = opencc.OpenCC("t2s.json")
        self._s2t_converter = opencc.OpenCC('s2t.json')

    def _split(self, text: str) -> List[str]:
        """Split long text into sentences with sentence-splitting punctuations.
        Parameters
        ----------
        text : str
            The input text.
        Returns
        -------
        List[str]
            Sentences.
        """
        text = self.SENTENCE_SPLITOR.sub(r'\1\n', text)
        text = text.strip()
        sentences = [sentence.strip() for sentence in re.split(r'\n+', text)]
        return sentences

    def _tranditional_to_simplified(self, text: str) -> str:
        return self._t2s_converter.convert(text)

    def _simplified_to_traditional(self, text: str) -> str:
        return self._s2t_converter.convert(text)

    def normalize_sentence(self, sentence):
        # basic character conversions
        sentence = self._tranditional_to_simplified(sentence)
        sentence = sentence.translate(F2H_ASCII_LETTERS).translate(
            F2H_DIGITS).translate(F2H_SPACE)

        # number related NSW verbalization
        sentence = RE_DATE.sub(replace_date, sentence)
        sentence = RE_DATE2.sub(replace_date2, sentence)
        sentence = RE_TIME.sub(replace_time, sentence)
        sentence = RE_TEMPERATURE.sub(replace_temperature, sentence)
        sentence = RE_RANGE.sub(replace_range, sentence)
        sentence = RE_FRAC.sub(replace_frac, sentence)
        sentence = RE_PERCENTAGE.sub(replace_percentage, sentence)
        sentence = RE_MOBILE_PHONE.sub(replace_phone, sentence)
        sentence = RE_TELEPHONE.sub(replace_phone, sentence)
        sentence = RE_DEFAULT_NUM.sub(replace_default_num, sentence)
        sentence = RE_NUMBER.sub(replace_number, sentence)

        return sentence

    def normalize(self, text):
        sentences = self._split(text)
        sentences = [self.normalize_sentence(sent) for sent in sentences]
        return sentences
