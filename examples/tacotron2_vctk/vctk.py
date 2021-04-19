from pathlib import Path
import pickle
import yaml

import numpy as np
from paddle.io import Dataset
from parakeet.frontend.vocab import Vocab
from parakeet.data import batch_spec, batch_text_id


class VCTK(Dataset):
    def __init__(self, root):
        root = Path(root).expanduser()
        self.root = root
        record_path = self.root / "metadata.pickle"
        self.wav_root = root / "wav"
        self.mel_root = root / "mel"
        with open(record_path, 'rb') as f:
            self.metadata = pickle.load(f)
        with open(self.root / "vocab" / "phonemes.yaml", 'rt') as f:
            phonemes = yaml.safe_load(f)
            self.phoneme_vocab = Vocab(phonemes)
            print(self.phoneme_vocab)
        with open(self.root / "vocab" / "speakers.yaml", 'rt') as f:
            speakers = yaml.safe_load(f)
            self.speaker_vocab = Vocab(speakers,
                                       padding_symbol=None,
                                       unk_symbol=None,
                                       start_symbol=None,
                                       end_symbol=None)

    def __getitem__(self, idx):
        metadatum = self.metadata[idx]
        fileid = metadatum['id']
        speaker_id = fileid.split('_')[0]
        s_id = self.speaker_vocab.lookup(speaker_id)
        phonemes = np.array([self.phoneme_vocab.lookup(item) \
                                for item in metadatum['phonemes']],
                            dtype=np.int64)
        mel_path = (self.mel_root / speaker_id / fileid).with_suffix(".npy")
        mel = np.load(mel_path).astype(np.float32)

        example = (phonemes, mel, s_id)
        return example

    def __len__(self):
        return len(self.metadata)


def collate_vctk_examples(examples):
    phonemes, mels, speaker_ids = list(zip(*examples))
    plens = np.array([item.shape[0] for item in phonemes], dtype=np.int64)
    slens = np.array([item.shape[1] for item in mels], dtype=np.int64)
    speaker_ids = np.array(speaker_ids, dtype=np.int64)

    phonemes, _ = batch_text_id(phonemes, pad_id=0)
    mels, _ = np.transpose(batch_spec(mels, pad_value=0.), [0, 2, 1])
    return phonemes, plens, mels, slens, speaker_ids
