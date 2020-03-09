# How to build your own model and experiment?

For a general deep learning experiment, there are 4 parts to care for.

1. process data to satisfy the need for training the model and iterate the them in batches;
2. define the model and the optimizer;
3. write the training process (including forward-backward computation, parameter update, logging, evaluation and other staff.)
4. configuration of the experiment.

## Data Processing

For processing data, `parakeet.data` provides flexible `Dataset`, `DataCargo`,`DataIterator`.

`DatasetMixin` and other datasets provide flexible Dataset API for building customized dataset.

`DataCargo`  is built from a dataset, but can be iterated in batches. We should provide `Sampler` and `batch function` in addition to build a `DataCargo`. `Sampler` specific which examples to pick, and `batch function` specifies how to batch. Commonly used Samplers are provides by `parakeet.data`.

 `DataIterator` is an iterator class for `DataCargo`.

We split data processing into two phases: example level processing and batching.

1. Sample level processing. This process is transforming an example into another example. This process can be defined in a `Dataset.get_example` or as a `transform` (callable object) and build a `TransformDataset` with it.

2. Batching. It is the processing of transforming a list of examples into a batch. The rationale is to transform an array of structures into a structure of arrays. We generally define a batch function (or a callable class).

To connect DataCargo with Paddlepaddle's asynchronous data loading mechanism, we need to create a `fluid.io.DataLoader` and connect it to the `Datacargo`.

The overview of data processing in an experiment with Parakeet is :

```text
Dataset --(transform)--> Dataset  --+
                         sampler  --|
                         batch_fn --+-> DataCargo --> DataLoader
```

The user need to define customized transform and batch function to accomplish this process. See [data](./data.md) for more details.

## Model

Parakeet provides commonly used functions, modules and models for the users to define their own models. Functions contains no trainable `Parameter`s, and is used in defining modules and models. Modules and modes are subclasses of `fluid.dygraph.Layer`. The distinction is that `module`s tend to be generic, simple and highly reusable, while `model`s tend to be task-sepcific, complicated and not that reusable. Some models are two complicated that we extract building blocks from it as separate classes but if they are not common and reusable enough, it is considered as a submodel.

In the structure of the project, modules are places in `parakeet.modules`, while models are in `parakeet.models` and group into folders like `waveflow` and `wavenet`, which include the whole model and thers submodels.

When developers want to add new models to `parakeet`, they can consider the distinctions described above and place code in appropriate place.



## Training Process

Training process is basically running a training loop for multiple times. A typical training loop consist of the procedures below:

1. Iterations over training datasets;
2. Prerocessing of mini batches;
3. Forward/backward computations of the neural networks;
4. Parameter updates;
5. Evaluations of the current parameters on validation datasets;
6. Logging intermediate results;
7. Saving checkpoint of the model and the optimizer.

In section `DataProcrssing` we have cover 1 and 2.

`Model` and `Optimizer` cover 3 and 4.

To keep the training loop clear, it's a good idea to define functions for save/load checkpoint, evaluation of validation set, logging and saving intermediate results. For some complicated model, it is also recommended to define a function to define the model. This function can be used in both train and inference, to ensure that the model is identical at training and inference.

Code is typically organized in this way:

```
├── configs          (example configuration)
├── data.py          (definition of custom Dataset, transform and batch function)
├── README.md        (README for the experiment)
├── synthesis.py     (code for inference)
├── train.py         (code for the training process)
└── utils.py         (all other utility functions)
```

## Configuration

Deep learning experiments have many options to configure. These configurations can be rougly group into different types: configurations about path of the dataset and path to save results, configurations about how to process data, configuration about the model and configurations about the training process.

Some configurations tend to change when running the code at different times. For example, path of the data and path to save results, whether to load model before training, etc. For these configurations, it's better to define them as command line arguments. We use `argparse` to handle them.

Other groups of configuration may overlap with others. For example, data processing and model may have some common configurations. The recommended way is to save them as configuration files, for example, `yaml` or `json`. We prefer `yaml`, for it is more human-reabable.



There are several examples in this repo, check the `Parakeet/examples` for more details. `Parakeet/examples` is where we place our experiments. Though experiments are not a part of package `parakeet`, it is a part of repo `Parakeet`. There are provided as examples and allow for the user to run our experiment out-of-the-box. Feel free to add new examples and contribute to `Parakeet`.
