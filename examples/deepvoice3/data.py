import numpy as np
import os
import csv
import pandas as pd

import paddle
from paddle import fluid
from paddle.fluid import dygraph as dg
from paddle.fluid.dataloader import Dataset, BatchSampler
from paddle.fluid.io import DataLoader

from parakeet.data import DatasetMixin, DataCargo, PartialyRandomizedSimilarTimeLengthSampler
from parakeet.g2p import en

class LJSpeech(DatasetMixin):
    def __init__(self, root):
        self._root = root
        self._table = pd.read_csv(
            os.path.join(root, "metadata.csv"), 
            sep="|", 
            encoding="utf-8", 
            quoting=csv.QUOTE_NONE, 
            header=None, 
            names=["num_frames", "spec_name", "mel_name", "text"],
            dtype={"num_frames": np.int64, "spec_name": str, "mel_name":str, "text":str})
    
    def num_frames(self):
        return self._table["num_frames"].to_list()

    def get_example(self, i):
        """
        spec (T_frame, C_spec)
        mel (T_frame, C_mel)
        """
        num_frames, spec_name, mel_name, text = self._table.iloc[i]
        spec = np.load(os.path.join(self._root, spec_name))
        mel = np.load(os.path.join(self._root, mel_name))
        return (text, spec, mel, num_frames)
    
    def __len__(self):
        return len(self._table)

class DataCollector(object):
    def __init__(self, p_pronunciation):
        self.p_pronunciation = p_pronunciation
        
    def __call__(self, examples):
        """
        output shape and dtype
        (B, T_text) int64
        (B,) int64
        (B, T_frame, C_spec) float32
        (B, T_frame, C_mel) float32
        (B,) int64
        """
        text_seqs = []
        specs = []
        mels = []
        num_frames = np.array([example[3] for example in examples], dtype=np.int64)
        max_frames = np.max(num_frames)

        for example in examples:
            text, spec, mel, _ = example
            text_seqs.append(en.text_to_sequence(text, self.p_pronunciation))
            # if max_frames - mel.shape[0] < 0:
            #     import pdb; pdb.set_trace()
            specs.append(np.pad(spec, [(0, max_frames - spec.shape[0]), (0, 0)]))
            mels.append(np.pad(mel, [(0, max_frames - mel.shape[0]), (0, 0)]))

        specs = np.stack(specs)
        mels = np.stack(mels)

        text_lengths = np.array([len(seq) for seq in text_seqs], dtype=np.int64)
        max_length = np.max(text_lengths)
        text_seqs = np.array([seq + [0] * (max_length - len(seq)) for seq in text_seqs], dtype=np.int64)
        return text_seqs, text_lengths, specs, mels, num_frames

if __name__ == "__main__":
    import argparse
    import tqdm
    import time
    from ruamel import yaml

    parser = argparse.ArgumentParser(description="load the preprocessed ljspeech dataset")
    parser.add_argument("--config", type=str, required=True, help="config file")
    parser.add_argument("--input", type=str, required=True, help="data path of the original data")
    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        config = yaml.safe_load(f)
    
    print("========= Command Line Arguments ========")
    for k, v in vars(args).items():
        print("{}: {}".format(k, v))
    print("=========== Configurations ==============")
    for k in ["p_pronunciation", "batch_size"]:
        print("{}: {}".format(k, config[k]))

    ljspeech = LJSpeech(args.input)
    collate_fn = DataCollector(config["p_pronunciation"])

    dg.enable_dygraph(fluid.CPUPlace())
    sampler = PartialyRandomizedSimilarTimeLengthSampler(ljspeech.num_frames())
    cargo = DataCargo(ljspeech, collate_fn, 
                      batch_size=config["batch_size"], sampler=sampler)
    loader = DataLoader\
           .from_generator(capacity=5, return_list=True)\
           .set_batch_generator(cargo)

    for i, batch in tqdm.tqdm(enumerate(loader)):
        continue
