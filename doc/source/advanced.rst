======================
Advanced Usage
======================

This sections covers how to extend parakeet by implementing you own models and 
experiments. Guidelines on implementation are also elaborated.

Model
-------------

As a common practice with paddlepaddle, models are implemented as subclasses
of ``paddle.nn.Layer``. More complicated models, it is recommended to split 
the model into different components.

For a encoder-decoder model, it is natural to split it into the encoder and 
the decoder. For a model composed of several similar layers, it is natural to 
extract the sublayer as a seperate layer.

There are two common ways to define a model which consists of several modules.

#. Define a module given the specifications.

   .. code-block:: python

      class MLP(nn.Layer):
          def __init__(self, input_size, hidden_size, output_size):
              self.linear1 = nn.Linear(input_size, hidden_size)
              self.linear2 = nn.Linear(hidden_size, output_size)
              
          def forward(self, x):
              return self.linear2(paddle.tanh(self.linear1(x))

      module = MLP(16, 32, 4) # intialize a module

   When the module is intended to be a generic reusable layer that can be 
   integrated into a larger model, we prefer to define it in this way.

   For considerations of readability and usability, we strongly recommend **NOT** to 
   pack specifications into a single object. Here's an example below.

   .. code-block:: python

      class MLP(nn.Layer):
          def __init__(self, hparams):
              self.linear1 = nn.Linear(hparams.input_size, hparams.hidden_size)
              self.linear2 = nn.Linear(hparams.hidden_size, hparams.output_size)
              
          def forward(self, x):
              return self.linear2(paddle.tanh(self.linear1(x))

   For a module defined in this way, it's harder for the user to initialize a 
   instance. The user have to read the code to check what attributes are used.

   Code in this style tend to pass a huge config object to initialize every 
   module used in an experiment, thought each module may not need the whole 
   configuration.
   
   We prefer to be explicit.

#. Define a module as a combination given its components.

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

   When a model is a complicated one made up of several components, each of which 
   has a separate functionality, and can be replaced by other components with the 
   same functionality, we prefer to define it in this way.

Data
-------------

Config
-------------

Experiment
--------------