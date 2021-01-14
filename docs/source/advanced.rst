======================
Advanced Usage
======================

This sections covers how to extend parakeet by implementing you own models and 
experiments. Guidelines on implementation are also elaborated.

Model
-------------

As a common practice with paddlepaddle, models are implemented as subclasses
of ``paddle.nn.Layer``. Models could be simple, like a single layer RNN. For 
complicated models, it is recommended to split the model into different 
components.

For a encoder-decoder model, it is natural to split it into the encoder and 
the decoder. For a model composed of several similar layers, it is natural to 
extract the sublayer as a separate layer.

There are two common ways to define a model which consists of several modules.

#. Define a module given the specifications. Here is an example with multilayer 
   perceptron.

   .. code-block:: python

      class MLP(nn.Layer):
          def __init__(self, input_size, hidden_size, output_size):
              self.linear1 = nn.Linear(input_size, hidden_size)
              self.linear2 = nn.Linear(hidden_size, output_size)
              
          def forward(self, x):
              return self.linear2(paddle.tanh(self.linear1(x))

      module = MLP(16, 32, 4) # intialize a module

   When the module is intended to be a generic and reusable layer that can be 
   integrated into a larger model, we prefer to define it in this way.

   For considerations of readability and usability, we strongly recommend 
   **NOT** to pack specifications into a single object. Here's an example below.

   .. code-block:: python

      class MLP(nn.Layer):
          def __init__(self, hparams):
              self.linear1 = nn.Linear(hparams.input_size, hparams.hidden_size)
              self.linear2 = nn.Linear(hparams.hidden_size, hparams.output_size)
              
          def forward(self, x):
              return self.linear2(paddle.tanh(self.linear1(x))

   For a module defined in this way, it's harder for the user to initialize an 
   instance. Users have to read the code to check what attributes are used.

   Also, code in this style tend to be abused by passing a huge config object 
   to initialize every module used in an experiment, thought each module may 
   not need the whole configuration.
   
   We prefer to be explicit.

#. Define a module as a combination given its components. Here is an example 
   for a sequence-to-sequence model.

   .. code-block:: python
   
      class Seq2Seq(nn.Layer):
          def __init__(self, encoder, decoder):
              self.encoder = encoder
              self.decoder = decoder
              
          def forward(self, x):
              encoder_output = self.encoder(x)
              output = self.decoder(encoder_output)
              return output
      
      encoder = Encoder(...)
      decoder = Decoder(...)
      model = Seq2Seq(encoder, decoder) # compose two components

   When a model is a complicated and made up of several components, each of which 
   has a separate functionality, and can be replaced by other components with the 
   same functionality, we prefer to define it in this way.

Data
-------------

Another critical componnet for a deep learning project is data. As a common 
practice, we use the dataset and dataloader abstraction. 

Dataset
^^^^^^^^^^
Dataset is the representation of a set of examples used for a projet. In most of 
the cases, dataset is a collection of examples. Dataset is an object which has 
methods below.

#. ``__len__``, to get the size of the dataset.
#. ``__getitem__``, to get an example by key or index.

Examples is a record consisting of several fields. In practice, we usually 
represent it as a namedtuple for convenience, yet dict and user-defined object 
are also supported.

We define our own dataset by subclassing ``paddle.io.Dataset``.

DataLoader
^^^^^^^^^^^
In deep learning practice, models are trained with minibatches. DataLoader 
meets the need for iterating the dataset in batches. It is done by providing 
a sampler and a batch function in addition to a dataset.

#. sampler, sample indices or keys used to get examples from the dataset.
#. batch function, transform a list of examples into a batch.

An commonly used sampler is ``RandomSampler``, it shuffles all the valid 
indices and then iterate over them sequentially. ``DistributedBatchSampler`` is 
a sampler used for distributed data parallel training, when the sampler handles 
data sharding in a dynamic way.

Batch function is used to transform selected examples into a batch. For a simple 
case where an example is composed of several fields, each of which is represented 
by an fixed size array, batch function can be simply stacking each field. For 
cases where variable size arrays are included in the example, batching could 
invlove padding and stacking. While in theory, batch function can do more like 
randomly slicing, etc.

For a custom dataset used for a custom model, it is required to define a batch 
function for it.

Config
-------------

It's common to change the running configuration to compare results. To keep track 
of running configuration, we use ``yaml`` configuration files.

Also, we want to interact with command line options. Some options that usually 
change according to running environments is provided by command line arguments. 
In addition, we wan to override an option in the config file without editing 
it. 

Taking these requirements in to consideration, we use `yacs <https://github.com/rbgirshick/yacs>`_ 
as a confi management tool. Other tools like `omegaconf <https://github.com/omry/omegaconf>`_ 
are also powerful and have similar functions.

In each example provided, there is a ``config.py``, where the default config is 
defined. If you want to get the default config, import ``config.py`` and call 
``get_cfg_defaults()`` to get the default config. Then it can be updated with 
yaml config file or command line arguments if needed.

For details about how to use yacs in experiments, see `yacs <https://github.com/rbgirshick/yacs>`_.


Experiment
--------------

