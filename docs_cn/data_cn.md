# 数据准备

本节主要讲述 `parakeet.data` 子模块的设计以及如何在实验中使用它。

`parakeet.data` 遵循 paddle 管用的数据准备流程。Dataset, Sampler, batch function, DataLoader.

## Dataset

我们假设数据集是样例的列表。你可以通过 `__len__` 方法获取其长度，并且可以通过 `__getitem__` 方法随机访问其元素。有了上述两个调节，我们也可以用 `iter(dataset)` 来获得一个 dataset 的迭代器。我们一般通过继承 `paddle.io.Dataset` 来创建自己的数据集。为其实现 `__len__` 方法和 `__getitem__` 方法即可。

出于数据处理，数据加载和数据集大小等方面的考虑，可以采用集中策略来调控数据集是否被懒惰地预处理，是否被懒惰地被加载，是否常驻内存等。

1. 数据在数据集实例化的时候被全部预处理并常驻内存。对于数据预处理比较快，且整个数据集较小的情况，可以采用这样的策略。因为整个的数据集的预处理在数据集实例化时完成，因此要求预处理很快，否则将要花时间等待数据集实例化。因为被处理后的数据集常驻内存，因此要求数据集较小，否则可能不能将整个数据集加载进内存。
2. 每个样例在被请求的时候预处理，并且把预处理的结果缓存。可以通过在数据集的 `__getitem__` 方法中调用单条样例的预处理方法来实现这个策略。这样做的条件一样是数据可以整个载入内存。但好处是不必花费很多时间等待数据集实例化。使用这个策略，则数据集被完整迭代一次之后，访问样例的时候会显著变快，因为不需要再次处理。但在首次使用的时候仍然会需要即时处理，所以如果快速评估数据迭代的数度还需要等数据集被迭代一遍。
3. 先将数据集预处理一遍把结果保存下来。再作为另一个数据集使用，这个新的数据集的 `__getitem__` 方法则只是从存储器读取数据。一般来说数据读取的性能并不会制约模型的训练，并且这也不要求内存必须足以装下整个数据集。是一种较为灵活的方法。但是会需要一个单独的预处理脚本，并且根据处理后的数据写一个数据集。

以上的三种只是一种概念上的划分，实际使用时候我们可能混用以上的策略。举例如下：

1. 对于一个样例的多个字段，有的是很小的，比如说文本，可能可能常驻内存；而对于音频，频谱或者图像，可能预先处理并存储，在访问时仅加载处理好的结果。
2. 对于某些比较大或者预处理比较慢的数据集。我们可以仅加载一个较小的元数据，里面包含了一些可以用于对样例进行排序或者筛选的特征码，则我们可以在不加载整个样例就可以利用这些元数据对数据进行排序或者筛选。

一般来说，我们将一个 Dataset 的子类看作是数据集和实验的具体需求之间的适配器。

parakeet 还提供了若干个高阶的 Dataset 类，用于从已有的 Dataset 产生新的 Dataset.

1. 用于字段组合的有 TupleDataset, DictDataset;
2. 用于数据集切分合并的有 SliceDataset, SubsetDataset, ChainDataset;
3. 用于缓存数据集的有 CacheDataset;
4. 用于数据集筛选的有 FilterDataset;
5. 用于变换数据集的有 TransformDataset.

可以灵活地使用这些高阶数据集来使数据处理更加灵活。

## DataLoader

`DataLoader` 类似 `Dataset` 也是可迭代对象，但是一般情况下，它是按批量来迭代的。在深度学习中我们需要 `DataLoader` 是因为把多个样例组成一个批次可以充分利用现代硬件的计算资源。可以根据一个 Dataset 构建一个 DataLoader，它可以被多次迭代。

构建 DataLoader 除了需要一个 Dataset 之外，还需要两个要素。

1. 如何组成批次。
2. 如何选取样例来组成批次；

下面的两个小节将分别提供这两个要素。

### batch function

批次是包含多个样例的列表经过某种变换的结果。假设一个样例是一个拥有多个字段的结构（在不同的编程语言可能有不同的实现，比如在 python 中可以是 tuple, dict 等，在 C/C++ 中可能是一个 struct）。那么包含多个样例的列表就是一个结构的阵列(array of structure, AOS). 而出于训练神经网络的需要，我们希望一个批次和一个样例一样，是拥有多个字段的一个结构。因此需要一个方法，把一个结构的阵列(array of structures)变成一个阵列的结构(structure of arrays).

下面是一个简单的例子：

下面的表格代表了两个样例，每个包含 5 个字段。

| weight | height | width | depth | density |
| ------ | ------ | ----- | ----- | ------- |
| 1.2    | 1.1    | 1.3   | 1.4   | 0.8     |
| 1.6    | 1.4    | 1.2   | 0.6   | 1.4     |

以上表格的 AOS 表示形式和 SOA 表示形式如下:

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

对于上述的例子，将 AOS 转换为 SOA 是平凡的。只要把所有样例的各个字段 stack 起来就可以。但事情并非总是如此简单。当一个字段包含一个序列，你可能就需要先把所有的序列都补长 (pad) 到最长的序列长度，然后才能把它们 stack 起来。对于某些情形，批次可能比样例多一些字段，比如说对于包含序列的样例，在补长之后，可能需要增设一个字段来记录那些字段的有效长度。因此，一般情况下，需要一个函数来实现这个功能，而且这是和这个数据集搭配的。当然除了函数之外，也可以使用任何的可调用对象，我们把这些称为 batch function.


### Sampler

有了 batch function（我们知道如何组成批次）, 接下来是另一个问题，将什么组成批次呢？当组建一个批次的时候，我们需要决定选取那些样例来组成它。因此我们预设数据集是可以随机访问的，我们只需要选取对应的索引即可。我们使用 sampler 来完成选取 index 的任务。

Sampler 被实现为产生整数的可迭代对象。假设数据集有 `N` 个样例，那么产生 `[0, N)` 之间的整数的迭代器就是一个合适的迭代器。最常用的 sampler 是 `SequentialSampler` 和 `RandomSampler`.

当迭代一个 DataLoader 的时候，首先 sampler 产生多个 index, 然后根据这些 index 去取出对应的样例，并调用 batch function 把这些样例组成一个批次。当然取出样例的过程是可并行的，但调用 batch function 组成 batch 不是。

另外的一种选择是使用 batch sampler, 它是产生整数列表的可迭代对象。对于一般的 sampler, 需要对其迭代器使用 next 多次才能产出多个 index, 而对于 batch sampler, 对其迭代器使用 next 一次就可以产出多个 index. 对于使用一般的 sampler 的情形，batch size 由 DataLoader 的来决定。而对于 batch sampler, 则是由它决定了 DataLoader 的 batch size, 因此可以用它来实现一些特别的需求，比如说动态 batch size.

## 示例代码

以下是我们使用 `parakeet.data` 处理 `LJSpeech` 数据集的代码。

首先，我们定义一个 class 来代表 LJspeech 数据集，它只是如其所是地加载了元数据，亦即数据集中的 `metadata.csv` 文件，其中记录了音频文件的文件名，以及转录文本。但并不加载音频，也并不做任何的预处理。我们有意让这个数据集保持简单，它仅需要数据集的路径来实例化。

```python
import csv
import numpy as np
import librosa
from pathlib import Path
from paddle.io import Dataset

from parakeet.data import batch_spec, batch_wav

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
```

然后我们定义一个 `Transform` 类，用于处理 `LJSpeechMetaData` 中的样例，将其转换为模型所需要的数据。对于不同的模型可以定义不同的 Transform，这样就可以共用 `LJSpeechMetaData` 的代码。

```python
from parakeet.audio import AudioProcessor
from parakeet.audio import LogMagnitude
from parakeet.frontend import English

class Transform(object):
    def __init__(self):
        self.frontend = English()
        self.processor = AudioProcessor(
            sample_rate=22050,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_max=8000)
        self.normalizer = LogMagnitude()

    def forward(self, record):
        fname, text, _ = meta_data:
        wav = processor.read_wav(fname)
        mel = processor.mel_spectrogram(wav)
        mel = normalizer.transform(mel)
        phonemes = frontend.phoneticize(text)
        ids = frontend.numericalize(phonemes)
        mel_name = os.path.splitext(os.path.basename(fname))[0]
        stop_probs = np.ones([mel.shape[1]], dtype=np.int64)
        stop_probs[-1] = 2
        return (ids, mel, stop_probs)
```

`Transform` 加载音频，并且提取频谱。把 `Transform` 实现为一个可调用的类可以方便地持有许多选项，比如和傅里叶变换相关的参数。这里可以把一个 `LJSpeechMetaData` 对象和一个 `Transform` 对象组合起来，创建一个 `TransformDataset`.

```python
from parakeet.data import TransformDataset

meta = LJSpeechMetaData(data_path)
transform = Transform()
ljspeech = TransformDataset(meta, transform)
```

当然也可以选择专门写一个转换脚本把转换后的数据集保存下来，然后再写一个适配的 Dataset 子类去加载这些保存的数据。实际这么做的效率会更高。

接下来我们需要写一个可调用对象将多个样例组成批次。因为其中的 ids 和 mel 频谱是序列数据，所以我们需要进行 padding.

```python
class LJSpeechCollector(object):
    """A simple callable to batch LJSpeech examples."""
    def __init__(self, padding_idx=0, padding_value=0.):
        self.padding_idx = padding_idx
        self.padding_value = padding_value

    def __call__(self, examples):
        ids = [example[0] for example in examples]
        mels = [example[1] for example in examples]
        stop_probs = [example[2] for example in examples]

        ids = batch_text_id(ids, pad_id=self.padding_idx)
        mels = batch_spec(mels, pad_value=self.padding_value)
        stop_probs = batch_text_id(stop_probs, pad_id=self.padding_idx)
        return ids, np.transpose(mels, [0, 2, 1]), stop_probs
```

以上的组件准备就绪后，可以准备整个数据流。

```python
def create_dataloader(source_path, valid_size, batch_size):
    lj = LJSpeechMeta(source_path)
    transform = Transform()
    lj = TransformDataset(lj, transform)

    valid_set, train_set = dataset.split(lj, valid_size)
    train_loader = DataLoader(
        train_set,
        return_list=False,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=LJSpeechCollector())
    valid_loader = DataLoader(
        valid_set,
        return_list=False,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=LJSpeechCollector())
    return train_loader, valid_loader
```

train_loader 和 valid_loader 可以被迭代。对其迭代器使用 next, 返回的是 `paddle.Tensor` 的 list, 代表一个 batch，这些就可以直接用作 `paddle.nn.Layer` 的输入了。
