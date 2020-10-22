from paddle.io import Dataset
from os import listdir
from os.path import splitext, join
from pathlib import Path
import librosa

class AudioFolderDataset(Dataset):
    def __init__(self, path, sample_rate, extension="wav"):
        self.root = path
        self.sample_rate = sample_rate
        self.extension = extension
        self.file_names = [join(self.root, x) for x in listdir(self.root) \
            if splitext(x)[-1] == self.extension]
        self.length = len(self.file_names)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        file_name = self.file_names[i]
        y, sr = librosa.load(file_name, sr=self.sample_rate) # pylint: disable=unused-variable
        return y


class LJSpeechMetaData(Dataset):
    def __init__(self, root):
        self.root = Path(root).expanduser()
        wav_dir = self.root / "wavs"
        csv_path = self.root / "metadata.csv"
        records = []
        speaker_name = "ljspeech"
        with open(str(csv_path), 'rt') as f:
            for line in f:
                filename, _, normalized_text = line.strip().split("|")
                filename = str(wav_dir / (filename + ".wav"))
                records.append([filename, normalized_text, speaker_name])
        self.records = records

    def __getitem__(self, i):
        return self.records[i]

    def __len__(self):
        return len(self.records)

