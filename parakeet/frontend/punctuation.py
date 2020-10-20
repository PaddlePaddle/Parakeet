import abc
import string

__all__ = ["get_punctuations"]

EN_PUNCT = [
    " ",
    "-",
    "...",
    ",",
    ".",
    "?",
    "!",
]

CN_PUNCT = [
    "、",
    "，",
    "；",
    "：",
    "。",
    "？",
    "！"
]

def get_punctuations(lang):
    if lang == "en":
        return EN_PUNCT
    elif lang == "cn":
        return CN_PUNCT
    else:
        raise ValueError(f"language {lang} Not supported")

