from pypinyin import pinyin, Style
from typing import List


def convert_to_pinyin(text: str) -> List[str]:
    """convert text into list of syllables, other characters that are not chinese, thus
    cannot be converted to pinyin are splited.
    """
    syllables = pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
    return syllables
    
