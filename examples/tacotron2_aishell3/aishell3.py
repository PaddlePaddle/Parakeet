from pathlib import Path
import numpy as np
import librosa
from paddle.io import Dataset
import pickle
from parakeet.frontend import Vocab
from parakeet.data import batch_text_id, batch_spec

from preprocess_transcription import _phones, _tones
voc_phones = Vocab(sorted(list(_phones)))
print(voc_phones)
voc_tones = Vocab(sorted(list(_tones)))
print(voc_tones)

# use yaml to store preprocessed aishell3 dataset
class AiShell3(Dataset):
    def __init__(self, root):
        self.root = Path(root).expanduser()
        self.embed_dir = self.root / "embed"
        self.mel_dir = self.root / "mel"

        with open (self.root / "metadata.pickle", 'rb') as f:
            self.records = pickle.load(f)
    
    def __getitem__(self, index):
        metadatum = self.records[index]
        sentence_id = metadatum["sentence_id"]
        speaker_id = sentence_id[:7]
        phones = metadatum["phones"]
        tones = metadatum["tones"]
        phones = np.array([voc_phones.lookup(item) for item in phones], dtype=np.int64)
        tones = np.array([voc_tones.lookup(item) for item in tones], dtype=np.int64)
        mel = np.load(str(self.mel_dir / speaker_id / (sentence_id + ".npy")))
        embed = np.load(str(self.embed_dir / speaker_id / (sentence_id + ".npy")))
        return phones, tones, mel, embed
    
    def __len__(self):
        return len(self.records)
    
def collate_aishell3_examples(examples):
    phones, tones, mel, embed = list(zip(*examples))

    text_lengths = np.array([item.shape[0] for item in phones], dtype=np.int64)
    spec_lengths = np.array([item.shape[1] for item in mel], dtype=np.int64)
    T_dec = np.max(spec_lengths)
    stop_tokens = (np.arange(T_dec) >= np.expand_dims(spec_lengths, -1)).astype(np.float32)
    phones, _ = batch_text_id(phones)
    tones, _ = batch_text_id(tones)
    mel, _ = batch_spec(mel)
    mel = np.transpose(mel, (0, 2, 1))
    embed = np.stack(embed)
    # 7 fields
    # (B, T), (B, T), (B, T, C), (B, C), (B,), (B,), (B, T)
    return phones, tones, mel, embed, text_lengths, spec_lengths, stop_tokens

if __name__ == "__main__":
    dataset = AiShell3("~/datasets/aishell3/train")
    example = dataset[0]

    examples = [dataset[i] for i in range(10)]
    batch = collate_aishell3_examples(examples)
    
    for field in batch:
        print(field.shape, field.dtype)
        
