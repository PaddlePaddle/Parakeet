# 安装

[TOC]


## 安装 PaddlePaddle

Parakeet 以 PaddlePaddle 作为其后端，因此依赖 PaddlePaddle，值得说明的是 Parakeet 要求 2.0 及以上版本的 PaddlePaddle。你可以通过 pip 安装。如果需要安装支持 gpu 版本的 PaddlePaddle，需要根据环境中的 cuda 和 cudnn 的版本来选择 wheel 包的版本。使用 conda 安装以及源码编译安装的方式请参考 [PaddlePaddle 快速安装](https://www.paddlepaddle.org.cn/install/quick/zh/2.0rc-linux-pip).

**gpu 版 PaddlePaddle**

```bash
python -m pip install paddlepaddle-gpu==2.0.0rc0.post101 -f https://paddlepaddle.org.cn/whl/stable.html
python -m pip install paddlepaddle-gpu==2.0.0rc0.post100 -f https://paddlepaddle.org.cn/whl/stable.html
```

**cpu 版 PaddlePaddle**

```bash
python -m pip install paddlepaddle==2.0.0rc0 -i https://mirror.baidu.com/pypi/simple
```

## 安装 libsndfile

因为 Parakeet 的实验中常常会需要用到和音频处理，以及频谱处理相关的功能，所以我们依赖 librosa 和 soundfile 进行音频处理。而 librosa 和 soundfile  依赖一个 C 的库 libsndfile, 因为这不是 python 的包，对于 windows 用户和 mac 用户，使用 pip 安装 soundfile 的时候，libsndfile 也会被安装。如果遇到问题也可以参考 [SoundFile](https://pypi.org/project/SoundFile).

对于 linux 用户，需要使用系统的包管理器安装这个包，常见发行版上的命令参考如下。


```bash
# ubuntu, debian
sudo apt-get install libsndfile1

# centos, fedora,
sudo yum install libsndfile

# openSUSE
sudo zypper in libsndfile
```

## 安装 Parakeet


我们提供两种方式来使用 Parakeet.

1. 需要运行 Parakeet 自带的实验代码，或者希望进行二次开发的用户，可以先从 github 克隆本工程，cd 仅工程目录，并进行可编辑式安装（不会被复制到 site-packages, 而且对工程的修改会立即生效，不需要重新安装），之后就可以使用了。

    ```bash
    # -e 表示可编辑式安装
    pip install -e .
    ```

2. 仅需要使用我们提供的训练好的模型进行预测，那么也可以直接安装 pypi 上的 wheel 包的版本。

    ```bash
    pip install paddle-parakeet
    ```
