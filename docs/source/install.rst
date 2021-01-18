=============
Installation
=============


Install PaddlePaddle
------------------------
Parakeet requires PaddlePaddle as its backend. Note that 2.0.0rc1 or newer versions
of paddle is required.

Since paddlepaddle has multiple packages depending on the device (cpu or gpu) 
and the dependency libraries, it is recommended to install a proper package of 
paddlepaddle with respect to the device and dependency library versons via 
pip. 

Installing paddlepaddle with conda or build paddlepaddle from source is also 
supported. Please refer to `PaddlePaddle installation <https://www.paddlepaddle.org.cn/install/quick/)>`_ for more details.

Example instruction to install paddlepaddle via pip is listed below.

**PaddlePaddle with gpu**

.. code-block:: bash

    python -m pip install paddlepaddle-gpu==2.0.0rc1.post101 -f https://paddlepaddle.org.cn/whl/stable.html
    python -m pip install paddlepaddle-gpu==2.0.0rc1.post100 -f https://paddlepaddle.org.cn/whl/stable.html


**PaddlePaddle with cpu**

.. code-block:: bash

    python -m pip install paddlepaddle==2.0.0rc1 -i https://mirror.baidu.com/pypi/simple


Install libsndfile
-------------------

Experimemts in parakeet often involve audio and spectrum processing, thus 
``librosa`` and ``soundfile`` are required. ``soundfile`` requires a extra 
C library ``libsndfile``, which is not always handled by pip.

For windows and mac users, ``libsndfile`` is also installed when installing
``soundfile`` via pip, but for linux users, installing ``libsndfile`` via
system package manager is required. Example commands for popular distributions 
are listed below.

.. code-block:: 

    # ubuntu, debian
    sudo apt-get install libsndfile1

    # centos, fedora
    sudo yum install libsndfile

    # openSUSE
    sudo zypper in libsndfile

For any problem with installtion of soundfile, please refer to 
`SoundFile <https://pypi.org/project/SoundFile>`_.

Install Parakeet
------------------

There are two ways to install parakeet according to the purpose of using it.

#. If you want to run experiments provided by parakeet or add new models and 
   experiments, it is recommended to clone the project from github 
   (`Parakeet <https://github.com/PaddlePaddle/Parakeet>`_), and install it in 
   editable mode.

   .. code-block:: bash
       
       git clone https://github.com/PaddlePaddle/Parakeet
       cd Parakeet
       pip install -e .

#. If you only need to use the models for inference by parakeet, install from
   pypi is recommended.

   .. code-block:: bash
   
       pip install paddle-parakeet
