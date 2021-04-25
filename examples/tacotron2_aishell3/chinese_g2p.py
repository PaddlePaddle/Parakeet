from chinese_text_to_pinyin import convert_to_pinyin
from chinese_phonology import convert, split_syllable
from typing import List, Tuple


def convert_sentence(text: str) -> List[Tuple[str]]:
    syllables = convert_to_pinyin(text)
    phones = []
    tones = []
    for syllable in syllables:
        p, t = split_syllable(syllable)
        phones.extend(p)
        tones.extend(t)

    return phones, tones
