import os
from pathlib import Path
import pickle
import numpy as np
import pandas
from paddle.io import Dataset, DataLoader

from parakeet.data.batch import batch_spec, batch_wav
from parakeet.data import dataset
from parakeet.audio import AudioProcessor

class LJSpeech(Dataset):
    """A simple dataset adaptor for the processed ljspeech dataset."""
    def __init__(self, root):
        self.root = Path(root).expanduser()
        meta_data = pandas.read_csv(
            str(self.root / "metadata.csv"),
            sep="\t",
            header=None,
            names=["fname", "frames", "samples"]
        )
        
        records = []
        for row in meta_data.itertuples() :
            mel_path = str(self.root / "mel" / (row.fname + ".npy"))
            wav_path = str(self.root / "wav" / (row.fname + ".npy"))
            records.append((mel_path, wav_path))
        self.records = records

    def __getitem__(self, i):
        mel_name, wav_name = self.records[i]
        mel = np.load(mel_name)
        wav = np.load(wav_name)
        return mel, wav

    def __len__(self):
        return len(self.records)


class LJSpeechCollector(object):
    """A simple callable to batch LJSpeech examples."""
    def __init__(self, padding_value=0.):
        self.padding_value = padding_value

    def __call__(self, examples):
        batch_size = len(examples)
        mels = [example[0] for example in examples]
        wavs = [example[1] for example in examples]
        mels = batch_spec(mels, pad_value=self.padding_value)
        wavs = batch_wav(wavs, pad_value=self.padding_value)
        audio_starts = np.zeros((batch_size,), dtype=np.int64)
        return mels, wavs, audio_starts


class LJSpeechClipCollector(object):
    def __init__(self, clip_frames=65, hop_length=256):
        self.clip_frames = clip_frames 
        self.hop_length = hop_length
    
    def __call__(self, examples):
        mels = []
        wavs = []
        starts = []
        for example in examples:
            mel, wav_clip, start = self.clip(example)
            mels.append(mel)
            wavs.append(wav_clip)
            starts.append(start)
        mels = batch_spec(mels)
        wavs = np.stack(wavs)
        starts = np.array(starts, dtype=np.int64)
        return mels, wavs, starts

    def clip(self, example):
        mel, wav = example
        frames = mel.shape[-1]
        start = np.random.randint(0, frames - self.clip_frames)
        wav_clip = wav[start * self.hop_length: (start + self.clip_frames) * self.hop_length]
        return mel, wav_clip, start


class DataCollector(object):
    def __init__(self,
                 context_size,
                 sample_rate,
                 hop_length,
                 train_clip_seconds,
                 valid=False):
        frames_per_second = sample_rate // hop_length
        train_clip_frames = int(
            np.ceil(train_clip_seconds * frames_per_second))
        context_frames = context_size // hop_length
        self.num_frames = train_clip_frames + context_frames

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.valid = valid

    def random_crop(self, sample):
        audio, mel_spectrogram = sample
        audio_frames = int(audio.size) // self.hop_length
        max_start_frame = audio_frames - self.num_frames
        assert max_start_frame >= 0, "audio is too short to be cropped"

        frame_start = np.random.randint(0, max_start_frame)
        # frame_start = 0  # norandom
        frame_end = frame_start + self.num_frames

        audio_start = frame_start * self.hop_length
        audio_end = frame_end * self.hop_length

        audio = audio[audio_start:audio_end]
        return audio, mel_spectrogram, audio_start

    def __call__(self, samples):
        # transform them first
        if self.valid:
            samples = [(audio, mel_spectrogram, 0)
                       for audio, mel_spectrogram in samples]
        else:
            samples = [self.random_crop(sample) for sample in samples]
        # batch them
        audios = [sample[0] for sample in samples]
        audio_starts = [sample[2] for sample in samples]
        mels = [sample[1] for sample in samples]

        mels = batch_spec(mels)

        if self.valid:
            audios = batch_wav(audios, dtype=np.float32)
        else:
            audios = np.array(audios, dtype=np.float32)
        audio_starts = np.array(audio_starts, dtype=np.int64)
        return audios, mels, audio_starts




