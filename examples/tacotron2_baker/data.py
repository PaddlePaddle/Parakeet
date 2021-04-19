from paddle.io import Dataset, DataLoader
from pathlib import Path
import yaml
import pickle
import numpy as np
from parakeet.frontend.vocab import Vocab
from parakeet.data.batch import batch_text_id, batch_spec
from preprocess_baker import _tones, _phones

voc_phones = Vocab(sorted(list(_phones)))
print(voc_phones)
voc_tones = Vocab(sorted(list(_tones)))
print(voc_tones)

class Baker(Dataset):
    def __init__(self, root):
        self.root = Path(root).expanduser()
        self.metadata_path = self.root / "metadata.pickle"
        self.mel_dir = self.root / "mel"
        with open(self.metadata_path,'rb') as f:
            self.metadata = pickle.load(f)

    def __getitem__(self, i):
        meta_datum = self.metadata[i]
        sentence_id = meta_datum['sentence_id']
        phones = ['<s>'] + meta_datum['phones'] + ['</s>']
        phones = np.array([voc_phones.lookup(item) for item in phones], dtype=np.int64)
        tones = ['<s>'] + meta_datum['tones'] + ['</s>']
        tones = np.array([voc_tones.lookup(item) for item in tones], dtype=np.int64)
        mel_path = (self.mel_dir / sentence_id).with_suffix(".npy")
        mel = np.load(mel_path)
        return phones, tones, mel
    
    def __len__(self):
        return len(self.metadata)

def collate_baker_examples(examples):
    phones, tones, mel = list(zip(*examples))
    text_lengths = np.array([item.shape[0] for item in phones], dtype=np.int64)
    spec_lengths = np.array([item.shape[1] for item in mel], dtype=np.int64)
    T_dec = np.max(spec_lengths)
    stop_tokens = (np.arange(T_dec) >= np.expand_dims(spec_lengths, -1)).astype(np.float32)
    phone, _ = batch_text_id(phones)
    tones, _ = batch_text_id(tones)
    mel, _ = batch_spec(mel)
    mel = np.transpose(mel, (0, 2, 1))
    
    return phones, tones, mel, text_lengths, spec_lengths, stop_tokens




if __name__ == "__main__":
    dataset = Baker("~/datasets/processed_BZNSYP")
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_baker_examples)
    it = iter(loader)
    batch = next(it)
    for item in batch:
        print(item.shape, item.dtype)
