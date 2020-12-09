# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from abc import ABC, abstractmethod
from typing import Union
from g2p_en import G2p
from g2pM import G2pM
import re
import unicodedata
from builtins import str as unicode
from parakeet.frontend import Vocab
from opencc import OpenCC
from parakeet.frontend.punctuation import get_punctuations
from parakeet.frontend.normalizer.numbers import normalize_numbers

__all__ = ["Phonetics", "English", "EnglishCharacter", "Chinese"]


class Phonetics(ABC):
    @abstractmethod
    def __call__(self, sentence):
        pass

    @abstractmethod
    def phoneticize(self, sentence):
        pass

    @abstractmethod
    def numericalize(self, phonemes):
        pass


class English(Phonetics):
    def __init__(self):
        self.backend = G2p()
        self.phonemes = list(self.backend.phonemes)
        self.punctuations = get_punctuations("en")
        self.vocab = Vocab(self.phonemes + self.punctuations)

    def phoneticize(self, sentence):
        start = self.vocab.start_symbol
        end = self.vocab.end_symbol
        phonemes = ([] if start is None else [start]) \
                 + self.backend(sentence) \
                 + ([] if end is None else [end])
        return phonemes

    def numericalize(self, phonemes):
        ids = [
            self.vocab.lookup(item) for item in phonemes
            if item in self.vocab.stoi
        ]
        return ids

    def reverse(self, ids):
        return [self.vocab.reverse(i) for i in ids]

    def __call__(self, sentence):
        return self.numericalize(self.phoneticize(sentence))

    @property
    def vocab_size(self):
        return len(self.vocab)


class EnglishCharacter(Phonetics):
    def __init__(self):
        self.backend = G2p()
        self.phonemes = list(self.backend.graphemes)
        self.punctuations = get_punctuations("en")
        self.vocab = Vocab(self.phonemes + self.punctuations)

    def _prepocessing(self, text):
        # preprocessing
        text = unicode(text)
        text = normalize_numbers(text)
        text = ''.join(
            char for char in unicodedata.normalize('NFD', text)
            if unicodedata.category(char) != 'Mn')  # Strip accents
        text = text.lower()
        text = re.sub(r"[^ a-z'.,?!\-]", "", text)
        text = text.replace("i.e.", "that is")
        text = text.replace("e.g.", "for example")
        return text

    def phoneticize(self, sentence):
        start = self.vocab.start_symbol
        end = self.vocab.end_symbol

        chars = ([] if start is None else [start]) \
                 + _prepocessing(sentence) \
                 + ([] if end is None else [end])
        return chars

    def numericalize(self, chars):
        ids = [
            self.vocab.lookup(item) for item in chars
            if item in self.vocab.stoi
        ]
        return ids

    def reverse(self, ids):
        return [self.vocab.reverse(i) for i in ids]

    def __call__(self, sentence):
        return self.numericalize(self.phoneticize(sentence))

    @property
    def vocab_size(self):
        return len(self.vocab)


class Chinese(Phonetics):
    def __init__(self):
        self.opencc_backend = OpenCC('t2s.json')
        self.backend = G2pM()
        self.phonemes = self._get_all_syllables()
        self.punctuations = get_punctuations("cn")
        self.vocab = Vocab(self.phonemes + self.punctuations)

    def _get_all_syllables(self):
        all_syllables = set([
            syllable for k, v in self.backend.cedict.items() for syllable in v
        ])
        return list(all_syllables)

    def phoneticize(self, sentence):
        simplified = self.opencc_backend.convert(sentence)
        phonemes = self.backend(simplified)
        start = self.vocab.start_symbol
        end = self.vocab.end_symbol
        phonemes = ([] if start is None else [start]) \
                 + phonemes \
                 + ([] if end is None else [end])
        return self._filter_symbols(phonemes)

    def _filter_symbols(self, phonemes):
        cleaned_phonemes = []
        for item in phonemes:
            if item in self.vocab.stoi:
                cleaned_phonemes.append(item)
            else:
                for char in item:
                    if char in self.vocab.stoi:
                        cleaned_phonemes.append(char)
        return cleaned_phonemes

    def numericalize(self, phonemes):
        ids = [self.vocab.lookup(item) for item in phonemes]
        return ids

    def __call__(self, sentence):
        return self.numericalize(self.phoneticize(sentence))

    @property
    def vocab_size(self):
        return len(self.vocab)

    def reverse(self, ids):
        return [self.vocab.reverse(i) for i in ids]
