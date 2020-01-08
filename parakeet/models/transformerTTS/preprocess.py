from pathlib import Path
import numpy as np
import pandas as pd
import librosa

from parakeet import g2p
from parakeet import audio

from parakeet.data.sampler import SequentialSampler, RandomSampler, BatchSampler
from parakeet.data.dataset import Dataset
from parakeet.data.datacargo import DataCargo
from parakeet.data.batch import TextIDBatcher, SpecBatcher

_ljspeech_processor = audio.AudioProcessor(
    sample_rate=22050, 
    num_mels=80, 
    min_level_db=-100, 
    ref_level_db=20, 
    n_fft=2048, 
    win_length= int(22050 * 0.05), 
    hop_length= int(22050 * 0.0125),
    power=1.2,
    preemphasis=0.97,
    signal_norm=True,
    symmetric_norm=False,
    max_norm=1.,
    mel_fmin=0,
    mel_fmax=None,
    clip_norm=True,
    griffin_lim_iters=60,
    do_trim_silence=False,
    sound_norm=False)

class LJSpeech(Dataset):
    def __init__(self, root):
        super(LJSpeech, self).__init__()
        assert isinstance(root, (str, Path)), "root should be a string or Path object"
        self.root = root if isinstance(root, Path) else Path(root)
        self.metadata = self._prepare_metadata()
        
    def _prepare_metadata(self):
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
        wav = _ljspeech_processor.load_wav(str(wav_path))
        mag = _ljspeech_processor.spectrogram(wav).astype(np.float32)
        mel = _ljspeech_processor.melspectrogram(wav).astype(np.float32)
        phonemes = np.array(g2p.en.text_to_sequence(normalized_text), dtype=np.int64)
        return (mag, mel, phonemes) # maybe we need to implement it as a map in the future

    def __getitem__(self, index):
        metadatum = self.metadata.iloc[index]
        example = self._get_example(metadatum)
        return example
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def __len__(self):
        return len(self.metadata)


def batch_examples(batch):
    texts = []
    mels = []
    mel_inputs = []
    text_lens = []
    pos_texts = []
    pos_mels = []
    for data in batch:
        _, mel, text = data
        mel_inputs.append(np.concatenate([np.zeros([mel.shape[0], 1], np.float32), mel[:,:-1]], axis=-1))
        text_lens.append(len(text))
        pos_texts.append(np.arange(1, len(text) + 1))
        pos_mels.append(np.arange(1, mel.shape[1] + 1))
        mels.append(mel)
        texts.append(text)
    
    # Sort by text_len in descending order
    texts = [i for i,_ in sorted(zip(texts, text_lens), key=lambda x: x[1], reverse=True)]
    mels = [i for i,_ in sorted(zip(mels, text_lens), key=lambda x: x[1], reverse=True)]
    mel_inputs = [i for i,_ in sorted(zip(mel_inputs, text_lens), key=lambda x: x[1], reverse=True)]
    pos_texts = [i for i,_ in sorted(zip(pos_texts, text_lens), key=lambda x: x[1], reverse=True)]
    pos_mels = [i for i,_ in sorted(zip(pos_mels, text_lens), key=lambda x: x[1], reverse=True)]
    text_lens = sorted(text_lens, reverse=True)

    # Pad sequence with largest len of the batch
    texts = TextIDBatcher(pad_id=0)(texts)
    pos_texts = TextIDBatcher(pad_id=0)(pos_texts)
    pos_mels = TextIDBatcher(pad_id=0)(pos_mels)
    mels = np.transpose(SpecBatcher(pad_value=0.)(mels), axes=(0,2,1))
    mel_inputs = np.transpose(SpecBatcher(pad_value=0.)(mel_inputs), axes=(0,2,1))
    return (texts, mels, mel_inputs, pos_texts, pos_mels, np.array(text_lens))

def batch_examples_vocoder(batch):
    mels=[]
    mags=[]
    for data in batch:
        mag, mel, _ = data
        mels.append(mel)
        mags.append(mag)

    mels = np.transpose(SpecBatcher(pad_value=0.)(mels), axes=(0,2,1))
    mags = np.transpose(SpecBatcher(pad_value=0.)(mags), axes=(0,2,1))

    return (mels, mags)


