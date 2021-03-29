import random

def cycle(iterable):
    # cycle('ABCD') --> A B C D A B C D A B C D ...
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        for element in saved:
            yield element

def random_cycle(iterable):
    # cycle('ABCD') --> A B C D B C D A A D B C ...
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    random.shuffle(saved)
    while saved:
        for element in saved:
            yield element
        random.shuffle(saved)