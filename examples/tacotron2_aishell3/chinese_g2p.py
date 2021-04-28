from typing import List, Tuple

from chinese_text_to_pinyin import convert_to_pinyin
from chinese_phonology import split_syllable


def convert_sentence(text: str) -> List[Tuple[str]]:
    """convert a sentence into two list: phones and tones"""
    syllables = convert_to_pinyin(text)
    phones = []
    tones = []
    for syllable in syllables:
        p, t = split_syllable(syllable)
        phones.extend(p)
        tones.extend(t)

    return phones, tones
