"""
A pinyin to phone transcription system for chinese.
Syllables are splited as initial and final. 'er' is also treated as s special symbol.
Tones are extracted and attached to finals.
"""
import re

# initials for mandarin chinese
# zero initials are not included
_initials = {
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh",
    "ch", "sh", "r", "z", "c", "s"
}

# finals for mandarin chines
# some symbols with different pronunciations are discriminated
# e.g. i -> {i, ii, iii}
# some symbols that are abbreviated are expanded
# e.g. iu -> iou, ui -> uei, un -> uen, bo -> b uo
# some symbols are transcripted according to zhuyin scheme
# e,g, in -> ien, ong -> ueng, iong -> veng
# üis  always replaced by v
_finals = {
    'ii',
    'iii',
    'a',
    'o',
    'e',
    'ea',
    'ai',
    'ei',
    'ao',
    'ou',
    'an',
    'en',
    'ang',
    'eng',
    'er',
    'i',
    'ia',
    'io',
    'ie',
    'iai',
    'iao',
    'iou',
    'ian',
    'ien',
    'iang',
    'ieng',
    'u',
    'ua',
    'uo',
    'uai',
    'uei',
    'uan',
    'uen',
    'uang',
    'ueng',
    'v',
    've',
    'van',
    'ven',
    'veng',
}

# Er hua symbol
# example tour2 -> phone: t ou &r, tone: 0 2 5
_ernized_symbol = {'&r'}

_specials = {'<pad>', '<unk>'}
_pauses = {"%",
           "$"}  # for different dataset, maybe you have to change this set

_phones = _initials | _finals | _ernized_symbol | _specials | _pauses

# 0: no tone, for initials
# {1, 2, 3, 4}: for tones in chinese
# 5: neutral tone
# <pad>: special token for padding
# <unk>: special token for unknown tone, though there will not be unknown tone
_tones = {'<pad>', '<unk>', '0', '1', '2', '3', '4', '5'}


def ernized(syllable):
    return syllable[:2] != "er" and syllable[-2] == 'r'


def convert(syllable):
    # expansion of o -> uo
    syllable = re.sub(r"([bpmf])o$", r"\1uo", syllable)
    # syllable = syllable.replace("bo", "buo").replace("po", "puo").replace("mo", "muo").replace("fo", "fuo")
    # expansion for iong, ong
    syllable = syllable.replace("iong", "veng").replace("ong", "ueng")

    # expansion for ing, in
    syllable = syllable.replace("ing", "ieng").replace("in", "ien")

    # expansion for un, ui, iu
    syllable = syllable.replace("un",
                                "uen").replace("ui",
                                               "uei").replace("iu", "iou")

    # rule for variants of i
    syllable = syllable.replace("zi", "zii").replace("ci", "cii").replace("si", "sii")\
        .replace("zhi", "zhiii").replace("chi", "chiii").replace("shi", "shiii")\
        .replace("ri", "riii")

    # rule for y preceding i, u
    syllable = syllable.replace("yi", "i").replace("yu", "v").replace("y", "i")

    # rule for w
    syllable = syllable.replace("wu", "u").replace("w", "u")

    # rule for v following j, q, x
    syllable = syllable.replace("ju", "jv").replace("qu",
                                                    "qv").replace("xu", "xv")

    return syllable


def split_syllable(syllable: str):
    if syllable in _pauses:
        # phone, tone
        return [syllable], ['0']

    tone = syllable[-1]
    syllable = convert(syllable[:-1])

    phones = []
    tones = []

    global _initials
    if syllable[:2] in _initials:
        phones.append(syllable[:2])
        tones.append('0')
        phones.append(syllable[2:])
        tones.append(tone)
    elif syllable[0] in _initials:
        phones.append(syllable[0])
        tones.append('0')
        phones.append(syllable[1:])
        tones.append(tone)
    else:
        phones.append(syllable)
        tones.append(tone)
    return phones, tones
