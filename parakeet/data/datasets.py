from paddle.io import Dataset

from os import listdir
from os.path import splitext, join
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