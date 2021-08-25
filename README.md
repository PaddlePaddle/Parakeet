# Parakeet

Parakeet aims to provide a flexible, efficient and state-of-the-art text-to-speech toolkit for the open-source community. It is built on PaddlePaddle Fluid dynamic graph and includes many influential TTS models proposed by [Baidu Research](http://research.baidu.com) and other research groups.  

<div align="center">
  <img src="images/logo.png" width=300 /> <br>
</div>

In particular, it features the latest [WaveFlow](https://arxiv.org/abs/1912.01219) model proposed by Baidu Research.

- WaveFlow can synthesize 22.05 kHz high-fidelity speech around 40x faster than real-time on a Nvidia V100 GPU without engineered inference kernels, which is faster than [WaveGlow](https://github.com/NVIDIA/waveglow) and serveral orders of magnitude faster than WaveNet.
- WaveFlow is a small-footprint flow-based model for raw audio. It has only 5.9M parameters, which is 15x smalller than WaveGlow (87.9M).
- WaveFlow is directly trained with maximum likelihood without probability density distillation and auxiliary losses as used in Parallel WaveNet and ClariNet, which simplifies the training pipeline and reduces the cost of development.

## Overview

In order to facilitate exploiting the existing TTS models directly and developing the new ones, Parakeet selects typical models and provides their reference implementations in PaddlePaddle. Further more, Parakeet abstracts the TTS pipeline and standardizes the procedure of data preprocessing, common modules sharing, model configuration, and the process of training and synthesis. The models supported here include Vocoders and end-to-end TTS models:

- Vocoders
  - [WaveFlow: A Compact Flow-based Model for Raw Audio](https://arxiv.org/abs/1912.01219)

- TTS models
  - [Neural Speech Synthesis with Transformer Network (Transformer TTS)](https://arxiv.org/abs/1809.08895)
  - [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](arxiv.org/abs/1712.05884)

## Updates

May-07-2021, Add an example for voice cloning in Chinese. Check [examples/tacotron2_aishell3](./examples/tacotron2_aishell3).


## Setup
It's difficult to install some dependent libraries for this repo in Windows system, we recommend that you **DO NOT** use Windows system, please use `Linux`.

Make sure the library `libsndfile1` is installed, e.g., on Ubuntu.

```bash
sudo apt-get install libsndfile1
```

### Install PaddlePaddle

See [install](https://www.paddlepaddle.org.cn/install/quick) for more details. This repo requires PaddlePaddle **2.1.2** or above.

### Install Parakeet
```bash
pip install -U paddle-parakeet
```

or
```bash
git clone https://github.com/PaddlePaddle/Parakeet
cd Parakeet
pip install -e .
```

If some python dependent packages cannot be installed successfully, you can run the following script first.
(replace `python3.6` with your own python version)
```bash
sudo apt install -y python3.6-dev
```

See [install](https://paddle-parakeet.readthedocs.io/en/latest/install.html) for more details.

## Examples

Entries to the introduction, and the launch of training and synthsis for different example models:

- [>>> WaveFlow](./examples/waveflow)
- [>>> Transformer TTS](./examples/transformer_tts)
- [>>> Tacotron2](./examples/tacotron2)
- [>>> Tacotron2_AISHELL3](./examples/tacotron2_aishell3)
- [>>> GE2E](./examples/ge2e)


## Audio samples

### TTS models (Acoustic Model + Neural Vocoder)

Check our [website](https://paddle-parakeet.readthedocs.io/en/latest/demo.html) for audio sampels.


## Checkpoints

### Tacotron2
1. [tacotron2_ljspeech_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_ljspeech_ckpt_0.3.zip)
2. [tacotron2_ljspeech_ckpt_0.3_alternative.zip](https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_ljspeech_ckpt_0.3_alternative.zip)

### Tacotron2_AISHELL3
1. [tacotron2_aishell3_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/tacotron2_aishell3_ckpt_0.3.zip)

### TransformerTTS
1. [transformer_tts_ljspeech_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_ckpt_0.3.zip)

### WaveFlow
1. [waveflow_ljspeech_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_ljspeech_ckpt_0.3.zip)

### GE2E
1. [ge2e_ckpt_0.3.zip](https://paddlespeech.bj.bcebos.com/Parakeet/ge2e_ckpt_0.3.zip)

## Copyright and License

Parakeet is provided under the [Apache-2.0 license](LICENSE).
