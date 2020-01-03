from pathlib import Path
import numpy as np
from paddle import fluid
from parakeet.data.sampler import DistributedSampler
from parakeet.data.datacargo import DataCargo
from preprocess import batch_examples, LJSpeech, batch_examples_vocoder

class LJSpeechLoader:
    def __init__(self, config, nranks, rank, is_vocoder=False):
        place = fluid.CUDAPlace(rank) if config.use_gpu else fluid.CPUPlace()

        LJSPEECH_ROOT = Path(config.data_path)
        dataset = LJSpeech(LJSPEECH_ROOT)
        sampler = DistributedSampler(len(dataset), nranks, rank)

        assert config.batch_size % nranks == 0
        each_bs = config.batch_size // nranks
        if is_vocoder:
            dataloader = DataCargo(dataset, sampler=sampler, batch_size=each_bs, shuffle=True, collate_fn=batch_examples_vocoder, drop_last=True)
        else:
            dataloader = DataCargo(dataset, sampler=sampler, batch_size=each_bs, shuffle=True, collate_fn=batch_examples, drop_last=True)
        
        self.reader = fluid.io.DataLoader.from_generator(
            capacity=32,
            iterable=True,
            use_double_buffer=True,
            return_list=True)
        self.reader.set_batch_generator(dataloader, place)
        
