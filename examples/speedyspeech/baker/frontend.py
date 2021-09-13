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
from typing import Dict
from typing import List

import numpy as np
import paddle

from parakeet.frontend.cn_frontend import Frontend as cnFrontend
from parakeet.frontend.vocab import Vocab


class Frontend():
    def __init__(self, phone_vocab_path=None, tone_vocab_path=None):
        self.frontend = cnFrontend()

        if phone_vocab_path:
            with open(phone_vocab_path, 'rt') as f:
                phones = [line.strip() for line in f.readlines()]
            self.vocab_phones = Vocab(
                phones, start_symbol=None, end_symbol=None)

        if tone_vocab_path:
            with open(tone_vocab_path, 'rt') as f:
                tones = [line.strip() for line in f.readlines()]
            self.vocab_tones = Vocab(tones, start_symbol=None, end_symbol=None)

    def _p2id(self, phonemes: List[str]) -> np.array:
        phone_ids = [self.vocab_phones.lookup(item) for item in phonemes]
        return np.array(phone_ids, np.int64)

    def _t2id(self, tones: List[str]) -> np.array:
        tone_ids = [self.vocab_tones.lookup(item) for item in tones]
        return np.array(tone_ids, np.int64)

    def _get_phone_tone(self, phonemes: List[str],
                        get_tone_ids: bool=False) -> List[List[str]]:
        phones = []
        tones = []
        if get_tone_ids and self.vocab_tones:
            for full_phone in phonemes:
                # split tone from finals
                match = re.match(r'^(\w+)([012345])$', full_phone)
                if match:
                    phone = match.group(1)
                    tone = match.group(2)
                    phones.append(phone)
                    tones.append(tone)
                else:
                    phones.append(full_phone)
                    tones.append('0')
        else:
            for phone in phonemes:
                phones.append(phone)
        return phones, tones

    def get_input_ids(
            self,
            sentence: str,
            merge_sentences: bool=True,
            get_tone_ids: bool=False) -> Dict[str, List[paddle.Tensor]]:
        phonemes = self.frontend.get_phonemes(
            sentence, merge_sentences=merge_sentences)
        result = {}
        phones = []
        tones = []
        temp_phone_ids = []
        temp_tone_ids = []
        for part_phonemes in phonemes:
            # add sil for speechspeech
            part_phonemes = ["sil"] + part_phonemes + ["sil"]
            phones, tones = self._get_phone_tone(
                part_phonemes, get_tone_ids=get_tone_ids)
            if tones:
                tone_ids = self._t2id(tones)
                tone_ids = paddle.to_tensor(tone_ids)
                temp_tone_ids.append(tone_ids)
            if phones:
                phone_ids = self._p2id(phones)
                phone_ids = paddle.to_tensor(phone_ids)
                temp_phone_ids.append(phone_ids)
        if temp_tone_ids:
            result["tone_ids"] = temp_tone_ids
        if temp_phone_ids:
            result["phone_ids"] = temp_phone_ids
        return result
