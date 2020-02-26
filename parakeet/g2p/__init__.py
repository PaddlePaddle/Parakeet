# coding: utf-8
"""Text processing frontend

All frontend module should have the following functions:

- text_to_sequence(text, p)
- sequence_to_text(sequence)

and the property:

- n_vocab

"""
from . import en

# optinoal Japanese frontend
try:
    from . import jp
except ImportError:
    jp = None

try:
    from . import ko
except ImportError:
    ko = None

# if you are going to use the frontend, you need to modify _characters in symbol.py:
# _characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? ' + '¡¿ñáéíóúÁÉÍÓÚÑ'
try:
    from . import es
except ImportError:
    es = None
