# 设计思想

## Dataset

### 非流式数据
非流式数据集就是一个 List，从 typing 的角度来说，就是说只要支持 `__getitem__` 就可以，亦即支持随机访问。这样子，我们可以把 Dataset 当作一个 List 来使用。

为了 scalability 我们希望数据的加载支持 **懒惰加载**。比如说，建立数据集的时候 (`__init__`)，只是做一些准备工作。比如只是获取了存有内容的文件名列表，或者只是加载了未经处理的数据，诸如此类。这些事情为获得真正的数据准备了条件，而且它们往往代价较低，所以可以一次记载进内存。

在此先考虑一种稍微简单的情形，这是一个 Sequence 数据集，并且我们可以得到一些 metadata。数据集需要支持一个方法，`__getitem__`，这个方法使得我们可以获取一条经过处理过的数据（参考 `Record`）。那么我们需要一个 Transform. 这个 Transform 从 metadata 中的一条得到一条处理过的数据。

另外，因为我们的数据集不是现成的数据，所以也不完全支持数据筛选，排序，根据相近长度做 partial sampling. 不过我们仍然有可能通过一些额外的步骤来得到一个 metadata, 用它来帮助我们排序，筛选和 bucketing. 但是这就依赖于在 prepare_metadata 做一些额外的事情，而且这也是因 metadata 而异的。

### 流式数据
另外，可能存在一种 Iterator 式的数据集，它不支持随机访问，也不能直接 shuffle。所以也不知道其长度，并且以后我们将会注意到，它也不能直接 shuffle, bucketing 也不是很方便。一般来说就要涉及到维持一个池，然后在池内 shuffle。我们目前对流式数据没有很好的支持。

## Example
经过处理后的一条记录，称为一个 `Example`，它类比的是表格数据中的一行。它可能包含几个字段的值，每一个值对应于这一行的值。而字段(参考 `Field`)就对应于表头。

我们可以将 Example 实现为简单的 Tuple, 或者 namedtuple，或者 dict, 或者可以 getattr 的对象。如果是出于简单的考虑，只要 tuple 就可以了。

值得注意的是，在这一套懒惰加载的实现中，我们并不是一开始创建数据集的时候就有 Example, 而是需要在数据真实加载的时候才有 Example. 而 torchtext 在这个事情上就和 torchvision, torchaudio 的数据集的思想相反，它先用 Field 把数据处理好了，然后只是把它们装进 dataset 里面。

因此，torchtext 是一个对文本数据做了特化的 dataset API, 对于其他的数据，这并不好用。

torchtext 这么做确实有它的考虑，比如说要先使用 Field 去创建构建出 Example (参考 `Example`)，然后才能用数据集构建字典（字典是字段的一部分，只是预处理的时候用不上它）。构建了字典之后，以后才能在 Batch 的时候顺便把 numericalize 先完成。所以这里事实上有点 hack, 在创建 Example 的时候，使用一个尚未完成的 Field, 只是很清楚，预处理的时候用不上字典，所以这是安全的。

而在迭代数据前必须先构建词典，而需要构建词典，一个比较常规的操作就是根据使用了这个字段的数据集本身去构建字典，听起来有点循环？是的。

另外，这一套流程其实和懒惰加载是矛盾的。如果使用来懒惰加载，甚至是流式数据处理，我们必须先得到 Vocab。如果 Vocab 是直接下载了加载那很好，但是如果必须根据数据集构建字典的时候，都来这么一次操作会比较麻烦。因为你构建字典本身就需要构建一次数据集，而这往往很慢。更好的操作是额外地预处理一番，并且把字典保存下来，以后就可以一直使用了。

## Field
Field 类比的是表格数据集中的表头。其实就日常的表格数据而言，表头也不仅仅是一个名字那么简单，它至少包含了类型信息，而类型本身就是一种 `callable`，也就是一种 Transform. 可以认为在未经 cast 之前，一个 Record 中所有字段的值都是字符串类型，经过表头的处理，才 cast 成了想要的类型。那么一个表格的表头可以视为一个 Transform, 将未经处理，但是结构和处理后的 Record 同构的 raw record 转换成了想要的 Record. 

torchtext 中的 Field 对象是一种包含了预处理方法和组 batch 方法的对象，dataset 中包含 Field. 使用 Field 的地方有两处，一处是 dataset 创建之前，用 Field 来预处理得到 Example, 另一处是在 batch 的时候使用它来组 Batch。

使用字段的一个优美之处在于，它把问题分解开来，你可以不必写一个巨大的 preprocessing 函数来处理把 raw record 转换为 record 这件事。而是将事情分解到每个字段上来。只要解决了字段的预处理，那么这些字段组合出来的 raw record 的预处理也能递归地得到解决。而且事情会变得更加可复用。比如说测试数据可能比训练数据少一个 label 字段，那么不用这种分解的方法，则预处理函数需要重写（虽然重写的时候里面也还是可以去逐个调用处理单个字段的函数）。

但是使用字段也存在一个比较强的预设，那就是 raw record 和 record 是同构的，record 有什么字段，raw record 也就应该有什么字段。但是实际情况未必都这么确切，比如说从 text 字段的值（句子）产生了文本 id 以及 长度，在 torchtext 中，这两个值组成的 tuple 才是这个字段的值。当两个值互相关联的时候，他们组合起来然后属于一个字段，其实也并非不可接受。但是当你打算这么做的时候就要开始谨慎了，因为这和（字段-值）的一一对应是违背的。

可能我们可以试图先构建一个和 record 同构的 raw record? 然而，事实是这并不合理。比如说，从音频提取频谱，其中一个 mel 频谱是由线性频谱进一步变换得到的，但是两个都比较重要，都需要提取。但是又不是各自独立地从音频读取，而是处于一套预处理流程的不同阶段或者分支。

因此在我们的设计中，`preprocess` 不应该强行要求按字段处理。而是由一个 `get_example` 方法来实现。因此组 batch 的方法也不一定必须要作为字段来实现，我们事实上只需要对一些符合特定要求的数据，实现对应的 collate function 就可以（参考 `collate_fn`）。

## DataLoader
首先需要说明的是，无论有没有 DataLoader, Dataset 本身就是可以迭代 `iter` 的。只是 DataLoader 需要支持更多的功能，比如批量化，动态批量化，比如打乱顺序，比如按照 Example 的长度排序，比如尽可能和长度相近的数据方式在一个 batch 里，比如支持放回抽样，加权抽样等等。迭代(`iter`） DataLoader 得到一个迭代器。然后就可以迭代这个迭代器了。

普通的顺序迭代数据集，以及 shuffle 之后再顺序迭代，都可以视为一个更加抽象的过程的示例，那就是抽样。只要有一个 batch_sampler，一次产生一个 batch 的 indices，就可以根据这些 indices 去数据集里面取出一个 `List[Record]`, 然后使用一个创建 batch 的方法创建 batch. 这个组 batch 的方法，在 torch 自身的 dataset API 中是作为 DataLoader 的 init 方法的一个参数，称为 `collate_fn: List[Example] -> Batch`。

在顺序过一遍，或者 shuffle 之后过一遍两种方法中，样本是不会重复的，整个数据集将会过一遍，这个过程中总共能够产生的抽样数是固定的。但是更广泛地说，我们也可以支持一种有放回的抽样，在这样的情景之下，我们没有明确的 epoch 概念，DataLoader 可以指定抽样次数，也可以支持永远地重复下去。

迭代 DataLoader, 返回一个 DataIterator. 这是一个真正的迭代器。因为为了是逻辑清晰，我们返回一个 Iterator 而不是用 yield 语句来写 DataLoader 的 `__iter__` 方法。

## Sampler
以上所说的抽样器(sampler), 每次产生一个 batch 的 indices。可以将它实现为一个 iterable 对象（实现了 `__iter__`）。迭代它，得到一个迭代器对象，每次需要的时候 next 一下就好。

可以实现一个单 index 的 sampler, 然后用一个 `batch_wrapper` 去包裹它。也可以直接传一个单 index sampler 然后通过 DataLoader 的功能来自动得产生一个 `batch_sampler`.

Sampler 只负责产生 index, 也可以将其实现为一个可以无限下去的对象，比如说就是一个 functionl, 每次 next 的时候产生一个值。

## collate_fn

组建 Batch 的一个函数。`collate_fn: List[Example] -> Batch`，一般情况下，我们可以预设 Batch 和 Example 是同构的，Example 有几个字段，Batch 就有几个字段。因此 collate_fn 事实上可以由多个字段对应的值的 collate_fn 组合出来。因此我们可以实现一些 collate function, 比如对于固定大小的 `array`, 只需要 `np.stack` 就可以。或者对于只有某一维度大小不定的 `ndarray`, 我们可以计算那一维度的 `max_len` 然后 使用 `np.pad` 补到这个长度，然后 `np.stack`. 对于不定长的 list, 就需要手动操作，一步步补足长度。对于复杂的 nested list (其实这就是 lod)，补长就比较复杂，可能就用户用到的时候自己定义了。

但是如果是文本数据，这里可能就需要 collate functor 而不是 collate function, 比如说 padding id 和 vocabuary 有关。而这个 vocabulary 也可能会用于预处理。至于为什么不能把 vocabulary, preprocess, collate_fn 组成字段，前面有不少的讨论，大约就是以上的原因。

## Transform 
一个 callable 对象，实现了 `__call__` 方法即可。在我们的设想中，因为有一些函数，它的配置实在太多，而输入和输出都是很直接了当的，本质上就是一个充分配置的 converter. 

我们可以组合出一个更大的 Transform.

## 总结

1. sampler 只要产生 index 就可以，只需要 dataset 不是流式 dataset. 比较独立。
2. example 只是 Plain Old Data, 如果出于简单的考虑，可以不做封装。
3. collate_fn 只是对符合某些特征的数据的组 batch 方式，比较独立。
4. dataset 目前主要考虑非流式数据，懒惰加载。只要数据集的长度是可以预先知道的，并且提供了一些 metadata, 那么只需要实现 `get_example` 方法（事实上是预处理），就可以制作 dataset。 基类不提供很多功能，这主要是因为语音、图像等数据集的处理都有比较多的自由，有比较多的可配置的地方。不像文本数据一样，虽然处理起来麻烦，但是流程一般比较固定,  tokenize 或者做一些  case 转换或者简繁转换之类。然后在组 batch 的时候数值化，pad, 组成 array。语音和图像的处理有较多可配置和自由定制的地方。因此，我们倾向于使用各种小 Transform 来组大的 Transform。
5. Transform 本质上就是 functor, 所以实现也比较独立。
6. dataloader 相对来说是最复杂的东西，它依赖 sampler 和 collate_fn。（多进程的支持，如果需要实现，其复杂性也主要在这里。）