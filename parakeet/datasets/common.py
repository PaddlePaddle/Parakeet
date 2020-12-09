from paddle.io import Dataset
import os
import librosa

__all__ = ["AudioFolderDataset"]

class AudioFolderDataset(Dataset):
    def __init__(self, path, sample_rate, extension="wav"):
        self.root = os.path.expanduser(path)
        self.sample_rate = sample_rate
        self.extension = extension
        self.file_names = [
            os.path.join(self.root, x) for x in os.listdir(self.root) \
                if os.path.splitext(x)[-1] == self.extension]
        self.length = len(self.file_names)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        file_name = self.file_names[i]
        y, _ = librosa.load(file_name, sr=self.sample_rate) # pylint: disable=unused-variable
        return y
