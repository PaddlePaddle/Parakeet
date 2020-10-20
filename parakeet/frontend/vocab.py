from typing import Dict, Iterable, List
from ruamel import yaml
from collections import OrderedDict

class Vocab(object):
    def __init__(self, symbols: Iterable[str], 
                  padding_symbol="<pad>",
                  unk_symbol="<unk>",
                  start_symbol="<s>",
                  end_symbol="</s>"):
        self.special_symbols = OrderedDict()
        for i, item in enumerate(
            [padding_symbol, unk_symbol, start_symbol, end_symbol]):
            if item:
                self.special_symbols[item] = len(self.special_symbols)
        
        self.padding_symbol = padding_symbol
        self.unk_symbol = unk_symbol
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
        
        
        self.stoi = OrderedDict()
        self.stoi.update(self.special_symbols)
        N = len(self.special_symbols)
        
        for i, s in enumerate(symbols):
            if s not in self.stoi:
                self.stoi[s] = N +i
        self.itos = {v: k for k, v in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)
    
    @property
    def num_specials(self):
        return len(self.special_symbols)

    # special tokens
    @property
    def padding_index(self):
        return self.stoi.get(self.padding_symbol, -1)

    @property
    def unk_index(self):
        return self.stoi.get(self.unk_symbol, -1)

    @property
    def start_index(self):
        return self.stoi.get(self.start_symbol, -1)

    @property
    def end_index(self):
        return self.stoi.get(self.end_symbol, -1)
    
    def __repr__(self):
        fmt = "Vocab(size: {},\nstoi:\n{})"
        return fmt.format(len(self), self.stoi)
    
    def __str__(self):
        return self.__repr__()
        
    def lookup(self, symbol):
        return self.stoi[symbol]
    
    def reverse(self, index):
        return self.itos[index]
    
    def add_symbol(self, symbol):
        if symbol in self.stoi:
            return 
        N = len(self.stoi)
        self.stoi[symbol] = N
        self.itos[N] = symbol
        
    def add_symbols(self, symbols):
        for symbol in symbols:
            self.add_symbol(symbol)
            
