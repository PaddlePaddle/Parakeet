from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from .. import g2p

from .sampler import SequentialSampler, RandomSampler, BatchSampler
from .dataset import Dataset
from .datacargo import DataCargo
from .batch import TextIDBatcher, SpecBatcher


class LJSpeech(Dataset):
    def __init__(self, root):
        super(LJSpeech, self).__init__()
        self.root = root
        self.metadata = self._prepare_metadata() # we can do this just for luck
        
    def _prepare_metadata(self):
        # if pure-stream case, each _prepare_metadata returns a generator
        csv_path = self.root.joinpath("metadata.csv")
        metadata = pd.read_csv(csv_path, sep="|", header=None, quoting=3,
                               names=["fname", "raw_text", "normalized_text"])
        return metadata
            
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

    def _batch_examples(self, minibatch):
        mag_batch = []
        mel_batch = []
        phoneme_batch = []
        for example in minibatch:
            mag, mel, phoneme = example
            mag_batch.append(mag)
            mel_batch.append(mel)
            phoneme_batch.append(phoneme)
        mag_batch = SpecBatcher(pad_value=0.)(mag_batch)
        mel_batch = SpecBatcher(pad_value=0.)(mel_batch)
        phoneme_batch = TextIDBatcher(pad_id=0)(phoneme_batch)
        return (mag_batch, mel_batch, phoneme_batch)

    def __getitem__(self, index):
        metadatum = self.metadata.iloc[index]
        example = self._get_example(metadatum)
        return example
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def __len__(self):
        return len(self.metadata)


