======================
Advanced Usage
======================

This sections covers how to extend parakeet by implementing you own models and 
experiments. Guidelines on implementation are also elaborated.

Model
-------------

As a common practice with paddlepaddle, models are implemented as subclasse
of ``paddle.nn.Layer``. More complicated models, it is recommended to split 
the model into different components.

For a encoder-decoder model, it is natural to split it into the encoder and 
the decoder. For a model composed of several similar layers, it is natural to 
extract the sublayer as a seperate layer.

There are two common ways to define a model which consists of several modules.

#. 


Data
-------------

Config
-------------

Experiment
--------------