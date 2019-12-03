import random

import librosa
import numpy as np
from paddle import fluid

import utils
from parakeet.datasets import ljspeech
from parakeet.data import dataset
from parakeet.data.sampler import DistributedSampler, BatchSampler
from parakeet.data.datacargo import DataCargo


class Dataset(ljspeech.LJSpeech):
    def __init__(self, config):
        super(Dataset, self).__init__(config.root)
        self.config = config
        self.fft_window_shift = config.fft_window_shift
        # Calculate context frames.
        frames_per_second = config.sample_rate // self.fft_window_shift
        train_clip_frames = int(np.ceil(
            config.train_clip_second * frames_per_second))
        context_frames = config.context_size // self.fft_window_shift
        self.num_frames = train_clip_frames + context_frames

    def _get_example(self, metadatum):
        fname, _, _ = metadatum
        wav_path = self.root.joinpath("wavs", fname + ".wav")

        config = self.config
        sr = config.sample_rate
        fft_window_shift = config.fft_window_shift
        fft_window_size = config.fft_window_size
        fft_size = config.fft_size
        
        audio, loaded_sr = librosa.load(wav_path, sr=None)
        assert loaded_sr == sr

        # Pad audio to the right size.
        frames = int(np.ceil(float(audio.size) / fft_window_shift))
        fft_padding = (fft_size - fft_window_shift) // 2
        desired_length = frames * fft_window_shift + fft_padding * 2
        pad_amount = (desired_length - audio.size) // 2
        
        if audio.size % 2 == 0:
            audio = np.pad(audio, (pad_amount, pad_amount), mode='reflect')
        else:
            audio = np.pad(audio, (pad_amount, pad_amount + 1), mode='reflect')
        
        # Normalize audio.
        audio = audio / np.abs(audio).max() * 0.999
        
        # Compute mel-spectrogram.
        # Turn center to False to prevent internal padding.
        spectrogram = librosa.core.stft(
            audio, hop_length=fft_window_shift,
            win_length=fft_window_size, n_fft=fft_size, center=False)
        spectrogram_magnitude = np.abs(spectrogram)
        
        # Compute mel-spectrograms.
        mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=fft_size,
                                              n_mels=config.mel_bands)
        mel_spectrogram = np.dot(mel_filter_bank, spectrogram_magnitude)
        mel_spectrogram = mel_spectrogram.T
        
        # Rescale mel_spectrogram.
        min_level, ref_level = 1e-5, 20
        mel_spectrogram = 20 * np.log10(np.maximum(min_level, mel_spectrogram))
        mel_spectrogram = mel_spectrogram - ref_level
        mel_spectrogram = np.clip((mel_spectrogram + 100) / 100, 0, 1)
        
        # Extract the center of audio that corresponds to mel spectrograms.
        audio = audio[fft_padding : -fft_padding]
        assert mel_spectrogram.shape[0] * fft_window_shift == audio.size

        return audio, mel_spectrogram


class Subset(dataset.Dataset): 
    def __init__(self, dataset, indices, valid):
        self.dataset = dataset
        self.indices = indices
        self.valid = valid

    def __getitem__(self, idx):
        fft_window_shift = self.dataset.fft_window_shift
        num_frames = self.dataset.num_frames
        audio, mel = self.dataset[self.indices[idx]]

        if self.valid:
            audio_start = 0
        else:
            # Randomly crop context + train_clip_second of audio.
            audio_frames = int(audio.size) // fft_window_shift
            max_start_frame = audio_frames - num_frames
            assert max_start_frame >= 0, "audio {} is too short".format(idx)

            frame_start = random.randint(0, max_start_frame)
            frame_end = frame_start + num_frames

            audio_start = frame_start * fft_window_shift
            audio_end = frame_end * fft_window_shift
            
            audio = audio[audio_start : audio_end]

        return audio, mel, audio_start

    def _batch_examples(self, batch):
        audios = [sample[0] for sample in batch]
        audio_starts = [sample[2] for sample in batch]
    
        # mels shape [num_frames, mel_bands]
        max_frames = max(sample[1].shape[0] for sample in batch) 
        mels = [utils.pad_to_size(sample[1], max_frames) for sample in batch]
        
        audios = np.array(audios, dtype=np.float32)
        mels = np.array(mels, dtype=np.float32)
        audio_starts = np.array(audio_starts, dtype=np.int32)
    
        return audios, mels, audio_starts

    def __len__(self):
        return len(self.indices)


class LJSpeech:
    def __init__(self, config, nranks, rank):
        place = fluid.CUDAPlace(rank) if config.use_gpu else fluid.CPUPlace()

        # Whole LJSpeech dataset.
        ds = Dataset(config)

        # Split into train and valid dataset.
        indices = list(range(len(ds)))
        train_indices = indices[config.valid_size:]
        valid_indices = indices[:config.valid_size]
        random.shuffle(train_indices)

        # Train dataset.
        trainset = Subset(ds, train_indices, valid=False)
        sampler = DistributedSampler(len(trainset), nranks, rank) 
        total_bs = config.batch_size
        assert total_bs % nranks == 0
        train_sampler = BatchSampler(sampler, total_bs // nranks,
            drop_last=True)
        trainloader = DataCargo(trainset, batch_sampler=train_sampler)

        trainreader = fluid.io.PyReader(capacity=50, return_list=True)
        trainreader.decorate_batch_generator(trainloader, place)
        self.trainloader = (data for _ in iter(int, 1)
            for data in trainreader())

        # Valid dataset.
        validset = Subset(ds, valid_indices, valid=True)
        # Currently only support batch_size = 1 for valid loader.
        validloader = DataCargo(validset, batch_size=1, shuffle=False)

        validreader = fluid.io.PyReader(capacity=20, return_list=True)
        validreader.decorate_batch_generator(validloader, place) 
        self.validloader = validreader
