# parakeet.data

This short guide shows the design of `parakeet.data` and how we use it in an experiment.

The most important concepts of `parakeet.data` are `DatasetMixin`, `DataCargo`, `Sampler`, `batch function` and `DataIterator`.

## Dataset

Dataset, as we assume here, is a list of examples. You can get its length by `len(dataset)`(which means its length is known, and we have to implement `__len__()` method for it). And you can access its items randomly by `dataset[i]`(which means we have to implement `__getitem__()` method for it). Furthermore,  you can iterate over it by `iter(dataset)` or `for example in dataset`, which means we have to implement `__iter__()` method for it.

### DatasetMixin

We provide an `DatasetMixin` object which provides the above methods. You can inherit `DatasetMixin` and implement `get_example()` method for it to define your own dataset class. The `get_example()` method is called by `__getitem__()` method automatically.

We also define several high-order Dataset classes, the obejcts of which can be built from some given Dataset objects.

### TupleDataset

Dataset that is a combination of several datasets of the same length. An example of a `Tupledataset` is a tuple of examples of its constituent datasets.

### DictDataset

Dataset that is a combination of several datasets of the same length. An example of the `Dictdataset` is a dict of examples of its constituent datasets.

### SliceDataset

`SliceDataset` is a slice of the base dataset.

### SubsetDataset

`SubsetDataset` is a subset of the base dataset.

### ChainDataset

`ChainDataset` is the concatenation of several datastes with the same fields.

### TransformDataset

A `TransformeDataset` is created by applying a `transform` to the examples of the base dataset. The `transform` is a callable object which takes an example of the base dataset as parameter and returns an example of the `TransformDataset`. The transformation is lazy, which means it is applied to an example only when requested.

### FilterDataset

A `FilterDataset` is created by applying a `filter` to the base dataset. A `filter` is a predicate that takes an example of the base dataset as parameter and returns a boolean. Only those examples that pass the filter are included in the `FilterDataset`.

Note that the filter is applied to all the examples in the base dataset when initializing a `FilterDataset`.

### CacheDataset

By default, we preprocess dataset lazily in `DatasetMixin.get_example()`. An example is preprocessed whenever requested. But `CacheDataset` caches the base dataset lazily, so each example is processed only once when it is first requested. When preprocessing the dataset is slow, you can use `Cachedataset` to speed it up, but caching may consume a lot of RAM if the dataset is large.

Finally, if preprocessing the dataset is slow and the processed dataset is too large to cache, you can write your own code to save them into files or databases, and then define a Dataset to load them. `Dataset` is flexible, so you can create your own dataset painlessly.

## DataCargo

`DataCargo`, like `Dataset`, is an iterable object, but it is an iterable oject of batches. We need `Datacargo` because in deep learning, batching examples into batches exploits the computational resources of modern hardwares. You can iterate over it by `iter(datacargo)` or `for batch in datacargo`. `DataCargo` is an iterable object but not an iterator, in that in can be iterated over more than once.

### batch function

The concept of a `batch` is something transformed from a list of examples. Assume that an example is a structure(tuple in python, or struct in C and C++) consists of several fields, then a list of examples is an array of structures(AOS, e.g. a dataset is an AOS). Then a batch here is a structure of arrays (SOA). Here is an example:

The table below represents 2 examples, each of which contains 5 fields.

| weight | height | width | depth | density |
| ------ | ------ | ----- | ----- | ------- |
| 1.2    | 1.1    | 1.3   | 1.4   | 0.8     |
| 1.6    | 1.4    | 1.2   | 0.6   | 1.4     |

The AOS representation and SOA representation of the table are shown below.

AOS:
```text
[(1.2, 1,1, 1,3, 1,4, 0.8),

 (1.6, 1.4, 1.2, 0.6, 1.4)]
```

SOA:
```text
([1,2, 1.6],
 [1.1, 1.4],
 [1.3, 1.2],
 [1.4, 0.6],
 [0.8, 1.4])
```

For the example above, converting an AOS to an SOA is trivial, just stacking every field for all the examples. But it is not always the case. When a field contains a sequence, you may have to pad all the sequences to the largest length then stack them together. In some other cases, we may want to add a field for the batch, for example, `valid_length` for each example. So in general, a function to transform an AOS to SOA is needed to build a `Datacargo` from a dataset. We call this the batch function (`batch_fn`), but you can use any callable object if you need to.

Usually we need to define the batch function as an callable object which stores all the options and configurations as its members. Its `__call__()` method transforms a list of examples into a batch.

### Sampler

Equipped with a batch function(we have known __how to batch__), here comes the next question. __What to batch?__ We need to decide which examples to pick when creating a batch. Since a dataset is a list of examples, we only need to pick indices for the corresponding examples. A sampler object is what we use to do this.

A `Sampler` is represented as an iterable object of integers. Assume the dataset has `N` examples, then an iterable object of intergers in the range`[0, N)` is an appropriate sampler for this dataset to build a `DataCargo`.

We provide several samplers that are ready to use, for example, `SequentialSampler` and `RandomSampler`.

## DataIterator

`DataIterator` is what returned by `iter(data_cargo)`. It can only be iterated over once.

Here's the analogy.

```text
Dataset   -->  Iterable[Example]  | iter(Dataset)    ->  Iterator[Example]
DataCargo -->  Iterable[Batch]    | iter(DataCargo)  ->  Iterator[Batch]
```

In order to construct an iterator of batches from an iterator of examples, we construct a DataCargo from a Dataset.



## Code Example

Here's an example of how we use `parakeet.data` to process the `LJSpeech` dataset with a wavenet model.

First, we would like to define a class which represents the LJSpeech dataset and loads it as-is. We try not to apply any preprocessings here.

```python
import csv
import numpy as np
import librosa
from pathlib import Path
import pandas as pd

from parakeet.data import DatasetMixin
from parakeet.data import batch_spec, batch_wav

class LJSpeechMetaData(DatasetMixin):
    def __init__(self, root):
        self.root = Path(root)
        self._wav_dir = self.root.joinpath("wavs")
        csv_path = self.root.joinpath("metadata.csv")
        self._table = pd.read_csv(
            csv_path,
            sep="|",
            header=None,
            quoting=csv.QUOTE_NONE,
            names=["fname", "raw_text", "normalized_text"])

    def get_example(self, i):
        fname, raw_text, normalized_text = self._table.iloc[i]
        fname = str(self._wav_dir.joinpath(fname + ".wav"))
        return fname, raw_text, normalized_text

    def __len__(self):
        return len(self._table)
```

We make this dataset simple in purpose. It requires only the path of the dataset, nothing more. It only loads the `metadata.csv` in the dataset when it is initialized, which includes file names of the audio files, and the transcriptions. We do not even load the audio files at `get_example()`.

Then we define a `Transform` object to transform an example of `LJSpeechMetaData` into an example we want for the model.

```python
class Transform(object):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __call__(self, example):
        wav_path, _, _ = example

        sr = self.sample_rate
        n_fft = self.n_fft
        win_length = self.win_length
        hop_length = self.hop_length
        n_mels = self.n_mels

        wav, loaded_sr = librosa.load(wav_path, sr=None)
        assert loaded_sr == sr, "sample rate does not match, resampling applied"

        # Pad audio to the right size.
        frames = int(np.ceil(float(wav.size) / hop_length))
        fft_padding = (n_fft - hop_length) // 2  # sound
        desired_length = frames * hop_length + fft_padding * 2
        pad_amount = (desired_length - wav.size) // 2

        if wav.size % 2 == 0:
            wav = np.pad(wav, (pad_amount, pad_amount), mode='reflect')
        else:
            wav = np.pad(wav, (pad_amount, pad_amount + 1), mode='reflect')

        # Normalize audio.
        wav = wav / np.abs(wav).max() * 0.999

        # Compute mel-spectrogram.
        # Turn center to False to prevent internal padding.
        spectrogram = librosa.core.stft(
            wav,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            center=False)
        spectrogram_magnitude = np.abs(spectrogram)

        # Compute mel-spectrograms.
        mel_filter_bank = librosa.filters.mel(sr=sr,
                                              n_fft=n_fft,
                                              n_mels=n_mels)
        mel_spectrogram = np.dot(mel_filter_bank, spectrogram_magnitude)
        mel_spectrogram = mel_spectrogram

        # Rescale mel_spectrogram.
        min_level, ref_level = 1e-5, 20  # hard code it
        mel_spectrogram = 20 * np.log10(np.maximum(min_level, mel_spectrogram))
        mel_spectrogram = mel_spectrogram - ref_level
        mel_spectrogram = np.clip((mel_spectrogram + 100) / 100, 0, 1)

        # Extract the center of audio that corresponds to mel spectrograms.
        audio = wav[fft_padding:-fft_padding]
        assert mel_spectrogram.shape[1] * hop_length == audio.size

        # there is no clipping here
        return audio, mel_spectrogram
```

`Transform` loads the audio files, and extracts `mel_spectrogram` from them. This transformation actually needs a lot of options to specify, namely, the sample rate of the audio files, the `n_fft`, `win_length`, `hop_length` of `stft` transformation, and `n_mels` for transforming spectrogram into mel_spectrogram. So we define it as a callable class. You can also use a closure, or a `partial` if you want to.

Then we defines a functor to batch examples into a batch. Because the two fields ( `audio` and `mel_spectrogram`) are both sequences, batching them is not trivial. Also, because the wavenet model trains in audio clips of a fixed length(0.5 seconds, for example), we have to truncate the audio when creating batches. We want to crop audio randomly when creating batches, instead of truncating them when preprocessing each example, because it allows for an audio to be truncated at different positions.

```python
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
```

When these 3 components are defined, we can start building our dataset with them.

```python
# building the ljspeech dataset
ljspeech_meta = LJSpeechMetaData(root)
transform = Transform(sample_rate, n_fft, win_length, hop_length, n_mels)
ljspeech = TransformDataset(ljspeech_meta, transform)

# split them into train and valid dataset
ljspeech_valid = SliceDataset(ljspeech, 0, valid_size)
ljspeech_train = SliceDataset(ljspeech, valid_size, len(ljspeech))

# building batch functions (they can be differnt for training and validation if you need it)
train_batch_fn = DataCollector(context_size, sample_rate, hop_length,
                               train_clip_seconds)
valid_batch_fn = DataCollector(
  context_size, sample_rate, hop_length, train_clip_seconds, valid=True)

# building the data cargo
train_cargo = DataCargo(
  ljspeech_train,
  train_batch_fn,
  batch_size,
  sampler=RandomSampler(ljspeech_train))

valid_cargo = DataCargo(
  ljspeech_valid,
  valid_batch_fn,
  batch_size=1, # only batch=1 for validation is enabled
  sampler=SequentialSampler(ljspeech_valid))
```

Here comes the next question, how to bring batches into Paddle's computation. Do we need some adapter to transform numpy.ndarray into Paddle's native Variable type? Yes.

First we can use `var = dg.to_variable(array)` to transform ndarray into Variable.

```python
for batch in train_cargo:
    audios, mels, audio_starts = batch
    audios = dg.to_variable(audios)
    mels = dg.to_variable(mels)
    audio_starts = dg.to_variable(audio_starts)

    # your training code here
```

In the code above, processing of the data and training of the model run in the same process. So the next batch starts to load after the training of the current batch has finished. There is actually better solutions for this. Data processing and model training can be run asynchronously. To accomplish this, we would use `DataLoader` from Paddle. This serves as an adapter to transform an iterable object of batches into another iterable object of batches, which runs asynchronously and transform each ndarray into `Variable`.

```python
# connect our data cargos with corresponding DataLoader
# now the data cargo is connected with paddle
with dg.guard(place):
    train_loader = fluid.io.DataLoader.from_generator(
        capacity=10,return_list=True).set_batch_generator(train_cargo, place)
   valid_loader = fluid.io.DataLoader.from_generator(
        capacity=10, return_list=True).set_batch_generator(valid_cargo, place)

    # iterate over the dataloader
    for batch in train_loader:
        audios, mels, audio_starts = batch
        # your trains cript here
```
