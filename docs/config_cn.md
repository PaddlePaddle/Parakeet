# 实验配置

本节主要讲述 parakeet 的推荐的配置实验的方式，以及我们做出这样的选择的原因。

## 配置选项的内容

深度学习实验常常有很多选项可配置。这些配置大概可以被分为几类：

1. 数据源以及数据处理方式配置；
2. 实验结果保存路径配置；
3. 数据预处理方式配置；
4. 模型结构和超参数配置；
5. 训练过程配置。

虽然这些配置之间也可能存在某些重叠项，比如数据预处理部分的配置可能就和模型配置有关。比如说 mel 频谱的维数，既可以理解为模型配置的一部分，也可以理解为数据处理配置的一部分。但大体上，配置文件是可以分成几个部分的。

## 常见配置文件格式

常见的配置文件的格式有 `ini`, `yaml`, `toml`, `json` 等。

`ini` 
优点：简单，支持字符串插值等操作。
缺点：仅支持两层结构，值不带类型信息，解析的时候需要手动 cast。

`yaml`
优点：格式简洁，值有类型，解析的时候一般不需手动 cast，支持写注释。
缺点：语法规范复杂。

`toml`
和 yaml 类似

`json`
优点：格式简单，
缺点：标记符号太多，可读性不佳，手写也容易出错。不支持注释。

出于语言本身的表达能力和可读性，我们选择 yaml, 但我们会尽可能使配置文件简单。

1. 类型上，只使用字符串，整数，浮点数，布尔值；
2. 结构嵌套上，尽可能只使用两层或更浅的结构。

## 配置选项和命令行参数处理

对于深度学习实验，有部分配置是经常会发生改变的，比如数据源以及保存实验结果的路径，或者加载的 checkpoint 的路径等。对于这些配置，更好的做法是把它们实现为命令行参数。

其余的不经常发生变动的参数，推荐将其写在配置文件中，我们推荐使用 `yaml` 作为配置文件，因为它允许添加注释，并且更加人类可读。

当然把所有的选项都有 argparse 来处理也可以，但是对于选项丰富的深度学习实验来说，都使用 argparse 会导致代码异常冗长。

但是需要注意的是，同时使用配置文件和命令行解析工具的时候，如果不做特殊处理，配置文件所支持的选项并不能显示在 argparse.ArgumentParser 的 usage 和 help 信息里。这主要是配置文件解析和 argparse 在设计上的一些固有的差异导致的。

通过一些手段把配置所支持的选项附加到 ArgumentParser 固然可以弥补这点，但是这会存在一些默认值的优先级哪一方更高的问题，是默认配置的优先级更高，比如还是 ArgumentParser 中的默认值优先级更高。

因此我们选择不把配置所支持的选项附加到 ArgumentParser，而是分开处理两部分。

## 实践

我们选择 yacs 搭配 argparse 作为配置解析工具，为 argparse 命令行新增一个选项 `--config` 来传入配置文件。yacs 有几个特点：

1. 支持 yaml 格式的配置文件（亦即支持配置层级嵌套以及有类型的值）；
2. 支持 config 的增量覆盖，以及由命令行参数覆盖配置文件等灵活的操作；
3. 支持 `.key` 递归访问属性，比字典式的 `["key"]` 方便；

我们推荐把默认的配置写成 python 代码（examples 中的每个例子都有一个 config.py，里面提供了默认的配置，并且带有注释）。而如果用户需要覆盖部分配置，则仅需要提供想要覆盖的部分配置即可，而不必提供一个完整的配置文件。这么做的考虑是：

1. 仅提供需要覆盖的选项也是许多软件配置的标准方式。
2. 对于同一个模型的两次实验，往往仅仅只有很少的配置发生变化，仅提供增量的配置比提供完整的配置更容易让用户看出两次实验的配置差异。
3. 运行脚本的时候可以不传 `--config` 参数，而以默认配置运行实验，简化运行脚本。

当新增实验的时候，可以参考 examples 里的例子来写默认配置文件。

除了可以通过 `--config` 命令行参数来指定用于覆盖的配置文件。另外，我们还可以通过新增一个 `--opts` 选项来接收 ArgumentParser 解析到的剩余命令行参数。这些参数将被用于进一步覆盖配置。使用方式是 `--opts key1 value1 key2 value2 ...`，即以空格分割键和值，比如`--opts training.lr 0.001 model.encoder_layers 4`。其中的键是配置中的键名，对于嵌套的选项，其键名以 `.` 连接。

## 默认的 ArgumentParser

我们提供了默认的 ArgumentParser（参考 `parakeet/training/cli.py`）, 它实现了上述的功能。它包含极简的命令行选项，只有 `--config`, `--data`, `--output`, `--checkpoint_path`, `--device`, `--nprocs` 和 `--opts` 选项。

这是一个深度学习基本都需要的一些命令行选项，因此当新增实验的时候，可以直接使用这个 ArgumentParser，当有超出这个范围的命令行选项时，也可以再继续新增。

1. `--config` 和 `--opts` 用于支持配置文件解析，而配置文件本身处理了每个实验特有的选项；
2. `--data` 和 `--output` 分别是数据集的路径和训练结果的保存路径（包含 checkpoints/ 文件夹，文本输出结果以及可视化输出结果）；
3. `--checkpoint_path` 用于在训练前加载某个 checkpoint, 当需要从某个特定的 checkpoint 加载继续训练。另外，在不传 `--checkpoint_path` 的情况下，如果 `--output` 下的 checkpoints/ 文件夹中包含了训练的结果，则默认会加载其中最新的 checkpoint 继续训练。
4. `--device` 和 `--nprocs` 指定了运行方式，`--device` 指定运行设备类型，是在 cpu 还是 gpu 上运行。`--nprocs` 指的是用多少个进程训练，如果 `nprocs` > 1 则意味着使用多进程并行训练。（注：目前只支持 gpu 多卡多进程训练。）

使用帮助信息如下:

```text
usage: train.py [-h] [--config FILE] [--data DATA_DIR] [--output OUTPUT_DIR]
                [--checkpoint_path CHECKPOINT_PATH] [--device {cpu,gpu}]
                [--nprocs NPROCS] [--opts ...]

optional arguments:
  -h, --help            show this help message and exit
  --config FILE         path of the config file to overwrite to default config
                        with.
  --data DATA_DIR       path to the datatset.
  --output OUTPUT_DIR   path to save checkpoint and log. If not provided, a
                        directory is created in runs/ to save outputs.
  --checkpoint_path CHECKPOINT_PATH
                        path of the checkpoint to load
  --device {cpu,gpu}    device type to use, cpu and gpu are supported.
  --nprocs NPROCS       number of parallel processes to use.
  --opts ...            options to overwrite --config file and the default
                        config, passing in KEY VALUE pairs
```








