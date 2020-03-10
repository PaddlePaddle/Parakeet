# How to build your own model and experiment?

For a general deep learning experiment, there are 4 parts to care for.

1. Preprocess dataset to meet the needs for model training and iterate over them in batches;
2. Define the model and the optimizer;
3. Write the training process (including forward-backward computation, parameter update, logging, evaluation, etc.)
4. Configure and launch the experiment.

## Data Processing

For processing data, `parakeet.data` provides `DatasetMixin`, `DataCargo` and `DataIterator`.

Dataset is an iterable object of examples. `DatasetMixin` provides the standard indexing interface, and other classes in [parakeet.data.dataset](../parakeet/data/dataset.py) provide flexible interfaces for building customized datasets.

`DataCargo` is an iterable object of batches. It differs from a dataset in that it can be iterated over in batches. In addition to a dataset, a `Sampler` and a `batch function` are required to build a `DataCargo`. `Sampler` specifies which examples to pick, and `batch function` specifies how to create a batch from them. Commonly used `Samplers` are provided by [parakeet.data](../parakeet/data/). Users should define a `batch function` for a datasets, in order to batch its examples.

 `DataIterator` is an iterator class for `DataCargo`. It is create when explicitly creating an iterator of a `DataCargo` by `iter(DataCargo)`, or iterating over a `DataCargo` with `for` loop.

Data processing is splited into two phases: sample-level processing and batching.

1. Sample-level processing. This process is transforming an example into another. This process can be defined as `get_example()` method of a dataset, or as a `transform` (callable object) and build a `TransformDataset` with it.

2. Batching. It is the process of transforming a list of examples into a batch. The rationale is to transform an array of structures into a structure of arrays. We generally define a batch function (or a callable object) to do this.

To connect a `DataCargo` with Paddlepaddle's asynchronous data loading mechanism, we need to create a `fluid.io.DataLoader` and connect it to the `Datacargo`.

The overview of data processing in an experiment with Parakeet is :

```text
Dataset --(transform)--> Dataset  --+
                         sampler  --+
                         batch_fn --+-> DataCargo --> DataLoader
```

The user need to define a customized transform and a batch function to accomplish this process. See [data](./data.md) for more details.

## Model

Parakeet provides commonly used functions, modules and models for the users to define their own models. Functions contain no trainable `Parameter`s, and are used in modules and models. Modules and modes are subclasses of `fluid.dygraph.Layer`. The distinction is that `module`s tend to be generic, simple and highly reusable, while `model`s tend to be task-sepcific, complicated and not that reusable. Some models are so complicated that we extract building blocks from it as separate classes but if these building blocks are not common and reusable enough, they are considered as submodels.

In the structure of the project, modules are placed in [parakeet.modules](../parakeet/modules/), while models are in [parakeet.models](../parakeet/models) and grouped into folders like `waveflow` and `wavenet`, which include the whole model and their submodels.

When developers want to add new models to `parakeet`, they can consider the distinctions described above and put the code in an appropriate place.



## Training Process

Training process is basically running a training loop for multiple times. A typical training loop consists of the procedures below:

1. Iterating over training dataset;
2. Prerocessing mini-batches;
3. Forward/backward computations of the neural networks;
4. Updating Parameters;
5. Evaluating the model on validation dataset;
6. Logging or saving intermediate results;
7. Saving checkpoints of the model and the optimizer.

In section `DataProcessing` we have cover 1 and 2.

`Model` and `Optimizer` cover 3 and 4.

To keep the training loop clear, it's a good idea to define functions for saving/loading of checkpoints, evaluation on validation set, logging and saving of intermediate results, etc. For some complicated model, it is also recommended to define a function to create the model. This function can be used in both train and inference, to ensure that the model is identical at training and inference.

Code is typically organized in this way:

```text
├── configs/         (example configuration)
├── data.py          (definition of custom Dataset, transform and batch function)
├── README.md        (README for the experiment)
├── synthesis.py     (code for inference)
├── train.py         (code for training)
└── utils.py         (all other utility functions)
```

## Configuration

Deep learning experiments have many options to configure. These configurations can be roughly grouped into different types: configurations about path of the dataset and path to save results, configurations about how to process data, configurations about the model and configurations about the training process.

Some configurations tend to change when running the code at different times, for example, path of the data and path to save results and whether to load model before training, etc. For these configurations, it's better to define them as command line arguments. We use `argparse` to handle them.

Other groups of configurations may overlap with others. For example, data processing and model may have some common options. The recommended way is to save them as configuration files, for example, `yaml` or `json`. We prefer `yaml`, for it is more human-reabable.



There are several examples in this repo, check [Parakeet/examples](../examples) for more details. `Parakeet/examples` is where we place our experiments. Though experiments are not a part of package `parakeet`, it is a part of repo `Parakeet`. They are provided as examples and allow for the users to run our experiment out-of-the-box. Feel free to add new examples and contribute to `Parakeet`.
