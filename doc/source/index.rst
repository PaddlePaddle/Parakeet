.. parakeet documentation master file, created by
   sphinx-quickstart on Thu Dec 17 20:01:34 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Parakeet 
====================================

``parakeet`` is a deep learning based text-to-speech toolkit built upon ``paddlepaddle`` framework. It aims to provide a flexible, efficient and state-of-the-art text-to-speech toolkit for the open-source community. It includes many influential TTS models proposed by `Baidu Research <http://research.baidu.com>`_ and other research groups. 

``parakeet`` mainly consists of components below.

#. Implementation of models and commonly used neural network layers.
#. Dataset abstraction and common data preprocessing pipelines.
#. Ready-to-run experiments.

.. toctree::
    :caption: Getting started
    :maxdepth: 1

    install
    tutorials

.. toctree::
    :caption: Design of Parakeet
    :maxdepth: 1
    
    advanced
    design

.. toctree::
    :caption: Documentation
    :maxdepth: 1

    parakeet.audio
    parakeet.data
    parakeet.datasets
    parakeet.frontend
    parakeet.modules
    parakeet.models
    parakeet.training
    parakeet.utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
