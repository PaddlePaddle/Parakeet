# coding: utf-8
from ..text.symbols import symbols
from ..text import sequence_to_text

import nltk
from random import random

n_vocab = len(symbols)


def text_to_sequence(text, p=0.0):
    from ..text import text_to_sequence
    text = text_to_sequence(text, ["basic_cleaners"])
    return text
