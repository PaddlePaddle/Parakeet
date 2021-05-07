from typing import List

from pypinyin import lazy_pinyin, Style


def convert_to_pinyin(text: str) -> List[str]:
    """convert text into list of syllables, other characters that are not chinese, thus
    cannot be converted to pinyin are splited.
    """
    syllables = lazy_pinyin(text,
                            style=Style.TONE3,
                            neutral_tone_with_five=True)
    return syllables
