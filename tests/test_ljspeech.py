from parakeet.datasets.ljspeech import LJSpeech
from parakeet.data.datacargo import DataCargo

from pathlib import Path

LJSPEECH_ROOT = Path("/workspace/datasets/LJSpeech-1.1")
ljspeech = LJSpeech(LJSPEECH_ROOT)
ljspeech_cargo = DataCargo(ljspeech, batch_size=16, shuffle=True)
for i, batch in enumerate(ljspeech_cargo):
    print(i)