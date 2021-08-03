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
from parakeet.frontend.cn_frontend import Frontend as cnFrontend


class Frontend():
    def __init__(self, phone_vocab_path=None, tone_vocab_path=None):
        self.frontend = cnFrontend()
        self.vocab_phones = {}
        self.vocab_tones = {}
        if phone_vocab_path:
            with open(phone_vocab_path, 'rt') as f:
                phn_id = [line.strip().split() for line in f.readlines()]
            for phn, id in phn_id:
                self.vocab_phones[phn] = int(id)
        if tone_vocab_path:
            with open(tone_vocab_path, 'rt') as f:
                tone_id = [line.strip().split() for line in f.readlines()]
            for tone, id in tone_id:
                self.vocab_tones[tone] = int(id)

    def _p2id(self, phonemes):
        # replace unk phone with sp
        phonemes = [
            phn if phn in self.vocab_phones else "sp" for phn in phonemes
        ]
        phone_ids = [self.vocab_phones[item] for item in phonemes]
        return np.array(phone_ids, np.int64)

    def _t2id(self, tones):
        # replace unk phone with sp
        tones = [
            tone if tone in self.vocab_tones else "0" for tone in tones
        ]
        tone_ids = [self.vocab_tones[item] for item in tones]
        return np.array(tone_ids, np.int64)

    def get_input_ids(self, sentence, get_tone_ids=False):
        phonemes = self.frontend.get_phonemes(sentence)
        result = {}
        phones = []
        tones = []
        if get_tone_ids and self.vocab_tones:
            for full_phone in phonemes:
                # split tone from finals
                match = re.match(r'^(\w+)([012345])$', full_phone)
                if match:
                    phones.append(match.group(1))
                    tones.append(match.group(2))
                else:
                    phones.append(full_phone)
                    tones.append('0')
            tone_ids = self._t2id(tones)
            tone_ids = paddle.to_tensor(tone_ids)
            result["tone_ids"] = tone_ids
        else:
            phones = phonemes
        phone_ids = self._p2id(phones)
        phone_ids = paddle.to_tensor(phone_ids)
        result["phone_ids"] = phone_ids
        return result
