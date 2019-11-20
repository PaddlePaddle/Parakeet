from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import g2p

from sampler import SequentialSampler, RandomSampler, BatchSampler
from dataset import Dataset
from dataloader import DataLoader

from collate import text_collate, spec_collate

LJSPEECH_ROOT = Path("/Users/chenfeiyu/projects/LJSpeech-1.1")
class LJSpeech(Dataset):
    def __init__(self, root=LJSPEECH_ROOT, lazy=True, stream=False):
        super(LJSpeech, self).__init__(lazy, stream)
        self.root = root
        self.metadata = self._prepare_metadata() # we can do this just for luck

        if self.stream:
            self.examples_generator = self._read() 
        
    def _prepare_metadata(self):
        # if pure-stream case, each _prepare_metadata returns a generator
        csv_path = self.root.joinpath("metadata.csv")
        metadata = pd.read_csv(csv_path, sep="|", header=None, quoting=3,
                               names=["fname", "raw_text", "normalized_text"])
        return metadata
    
    def _read(self):
        for _, metadatum in self.metadata.iterrows():
            example = self._get_example(metadatum)
            yield example
            
    def _get_example(self, metadatum):
        """All the code for generating an Example from a metadatum. If you want a 
        different preprocessing pipeline, you can override this method. 
        This method may require several processor, each of which has a lot of options.
        In this case, you'd better pass a composed transform and pass it to the init
        method.
        """
        
        fname, raw_text, normalized_text = metadatum
        wav_path = self.root.joinpath("wavs", fname + ".wav")
        
        # load -> trim -> preemphasis -> stft -> magnitude -> mel_scale -> logscale -> normalize
        wav, sample_rate = librosa.load(wav_path, sr=None) # we would rather use functor to hold its parameters
        trimed, _ = librosa.effects.trim(wav)
        preemphasized = librosa.effects.preemphasis(trimed)
        D = librosa.stft(preemphasized)
        mag, phase = librosa.magphase(D)
        mel = librosa.feature.melspectrogram(S=mag)
        
        mag = librosa.amplitude_to_db(S=mag)
        mel = librosa.amplitude_to_db(S=mel)
        
        ref_db = 20
        max_db = 100
        mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
        mel = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

        phonemes = np.array(g2p.en.text_to_sequence(normalized_text), dtype=np.int64)
        return (mag, mel, phonemes) # maybe we need to implement it as a map in the future

    def __getitem__(self, index):
        if self.stream:
            raise ValueError("__getitem__ is invalid in stream mode")
        metadatum = self.metadata.iloc[index]
        example = self._get_example(metadatum)
        return example
    
    def __iter__(self):
        if self.stream:
            for example in self.examples_generator:
                yield example
        else:
            for i in range(len(self)):
                yield self[i]
    
    def __len__(self):
        if self.stream:
            raise ValueError("__len__ is invalid in stream mode")
        return len(self.metadata)


def fn(minibatch):
    mag_batch = []
    mel_batch = []
    phoneme_batch = []
    for example in minibatch:
        mag, mel, phoneme = example
        mag_batch.append(mag)
        mel_batch.append(mel)
        phoneme_batch.append(phoneme)
    mag_batch = spec_collate(mag_batch)
    mel_batch = spec_collate(mel_batch)
    phoneme_batch = text_collate(phoneme_batch)
    return (mag_batch, mel_batch, phoneme_batch)

if __name__ == "__main__":
    ljspeech = LJSpeech(LJSPEECH_ROOT)
    ljspeech_loader = DataLoader(ljspeech, batch_size=16, shuffle=True, collate_fn=fn)
    for i, batch in enumerate(ljspeech_loader):
        print(i)

