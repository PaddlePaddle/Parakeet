# 实验流程

实验中有不少细节需要注意，比如模型的保存和加载，定期进行验证，文本 log 和 可视化 log，保存配置文件等，另外对于不同的运行方式还有额外的处理，这些代码可能比较繁琐，但是对于追踪代码变化对结果的影响以及 debug 都非常重要。为了减少写这部分代码的成本，我们提供了不少通用的辅助代码，比如用于保存和加载，以及可视化的代码，可供实验代码直接使用。

而对于整个实验过程，我们提供了一个 ExperimentBase 类，它是在模型和实验开发的过程抽象出来的训练过程模板，可以作为具体实验的基类使用。相比 chainer 中的 Trainer 以及 keras 中的 Model.fit 而言，ExperimentBase 是一个相对低层级的 API。它是作为基类来使用，用户仍然需要实现整个训练过程，也因此可以自由控制许多东西；而不是作为一种组合方式来使用，用户只需要提供模型，数据集，评价指标等就能自动完成整个训练过程。

前者的方式并不能节省很多代码量，只是以一种标准化的方式来组织代码。后者的方式虽然能够节省许多代码量，但是把如何组成整个训练过程的方式对用户隐藏了。如果需要为标准的训练过程添加一些自定义行为，则必须通过 extension/hook 等方式来实现，在一些固定的时点加入一些自定义行为（比如 iteration 开始、结束时，epoch 开始、结束时，整个训练流程开始、结束时）。

通过 extension/hook 之类的方式来为训练流程加入自定义行为，往往存在一些 access 的限制。extension/hook 一般是通过 callable 的形式来实现，但是这个 callable 可访问的变量往往是有限的，比如说只能访问 model, optimzier, dataloader, iteration, epoch, metric 等，如果需要访问其他的中间变量，则往往比较麻烦。

此外，组合式的使用方式往往对几个组件之间传输数据的协议有一些预设。一个常见的预设是：dataloader 产生的 batch 即是 model 的输入。在简单的情况下，这样大抵是没有问题的，但是也存在一些可能，模型需要除了 batch 之外的输入。令一个常见的预设是：criterion 仅需要 model 的 input 和 output 就能计算 loss, 但这么做其实存在 overkill 的可能，某些情况下，不需要 input 和 output 的全部字段就能计算 loss，如果为了满足协议而把 criterion 的接口设计成一样的，存在输出不必要的参数的问题。

## ExperimentBase 的设计

因此我们选择了低层次的接口，用户仍然可以自由操作训练过程，而只是对训练过程做了粗粒度的抽象。可以参考 [ExperimentBase](parakeet/training/experiment.py) 的代码。

继承 ExperimentBase 写作自己的实验类的时候，需要遵循一下的一些规范：

1. 包含 `.model`, `.optimizer`, `.train_loader`, `.valid_loader`, `.config`, `.args` 等属性。
2. 配置需要包含一个 `.training` 字段, 其中包含 `valid_interval`, `save_interval` 和 `max_iteration` 几个键. 它们被用作触发验证，保存 checkpoint 以及停止训练的条件。
3. 需要实现四个方法 `train_batch`, `valid`, `setup_model` and `setup_dataloader`。`train_batch` 是在一个 batch 的过程，`valid` 是在整个验证数据集上执行一次验证的过程，`setup_model` 是初始化 model 和 optimizer 的过程，其他的模型构建相关的代码也可以放在这里，`setup_dataloader` 是 train_loader 和 valid_loader 的构建过程。

实验的初始化过程如下, 包含了创建模型，优化器，数据迭代器，准备输出目录，logger 和可视化，保存配置的工作，除了 `setup_dataloader` 和 `self.setup_model` 需要自行实现，其他的几个方法都已有标准的实现。

```python
def __init__(self, config, args):
    self.config = config
    self.args = args

def setup(self):
    paddle.set_device(self.args.device)
    if self.parallel:
        self.init_parallel()

    self.setup_output_dir()
    self.dump_config()
    self.setup_visualizer()
    self.setup_logger()
    self.setup_checkpointer()

    self.setup_dataloader()
    self.setup_model()

    self.iteration = 0
    self.epoch = 0
```

使用的时候只要一下的代码即可配置好一次实验：

```python
exp = Experiment(config, args)
exp.setup()
```

整个训练流程可以表示如下:

```python
def train(self):
    self.new_epoch()
    while self.iteration < self.config.training.max_iteration:
        self.iteration += 1
        self.train_batch()

        if self.iteration % self.config.training.valid_interval == 0:
            self.valid()

        if self.iteration % self.config.training.save_interval == 0:
            self.save()
```

使用时只需要执行如下代码即可开始实验。

```python
exp.run()
```
