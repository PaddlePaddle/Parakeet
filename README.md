# Parakeet

Parakeet aims to provide a flexible, efficient and state-of-the-art text-to-speech toolkit for the open-source community. It is built on PaddlePaddle Fluid dynamic graph and includes many influential TTS models proposed by [Baidu Research](http://research.baidu.com) and other research groups.  

<div align="center">
  <img src="images/logo.png" width=450 /> <br>
</div>

In particular, it features the latest [WaveFlow] (https://arxiv.org/abs/1912.01219) model proposed by Baidu Research.

- WaveFlow can synthesize 22.05 kHz high-fidelity speech around 40x faster than real-time on a Nvidia V100 GPU without engineered inference kernels, which is faster than [WaveGlow] (https://github.com/NVIDIA/waveglow) and serveral orders of magnitude faster than WaveNet.
- WaveFlow is a small-footprint flow-based model for raw audio. It has only 5.9M parameters, which is 15x smalller than WaveGlow (87.9M) and comparable to WaveNet (4.6M).
- WaveFlow is directly trained with maximum likelihood without probability density distillation and auxiliary losses as used in Parallel WaveNet and ClariNet, which simplifies the training pipeline and reduces the cost of development.

## Overview

In order to facilitate exploiting the existing TTS models directly and developing the new ones, Parakeet selects typical models and provides their reference implementations in PaddlePaddle. Further more, Parakeet abstracts the TTS pipeline and standardizes the procedure of data preprocessing, common modules sharing, model configuration, and the process of training and synthesis. The models supported here includes Vocoders and end-to-end TTS models:

- Vocoders
  - [WaveFlow: A Compact Flow-based Model for Raw Audio](https://arxiv.org/abs/1912.01219)
  - [ClariNet: Parallel Wave Generation in End-to-End Text-to-Speech](https://arxiv.org/abs/1807.07281)
  - [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)

- TTS models
  - [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654)
  - [Neural Speech Synthesis with Transformer Network (Transformer TTS)](https://arxiv.org/abs/1809.08895)
  - [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263)

And more will be added in the future.

See the guide for details about how to build your own model and experiment in Parakeet.

## Setup

Make sure the library `libsndfile1` is installed, e.g., on Ubuntu.

```bash
sudo apt-get install libsndfile1
```

### Install PaddlePaddle

See [install](https://www.paddlepaddle.org.cn/install/quick) for more details. This repo requires PaddlePaddle 1.7.0 or above.

### Install Parakeet

```bash
git clone https://github.com/PaddlePaddle/Parakeet
cd Parakeet
pip install -e .
```

### Install CMUdict for nltk

CMUdict from nltk is used to transform text into phonemes.

```python
import nltk
nltk.download("punkt")
nltk.download("cmudict")
```




## Examples

Entries to the introduction, and the launch of training and synthsis for different example models:


- [>>> WaveFlow](./examples/waveflow)
- [>>> Clarinet](./examples/clarinet)
- [>>> WaveNet](./examples/wavenet)
- [>>> Deep Voice 3](./examples/deepvoice3)
- [>>> Transformer TTS](./examples/transformer_tts)
- [>>> FastSpeech](./examples/fastspeech)


## Pre-trained models and audio samples

Parakeet also releases some well-trained parameters for the example models, which can be accessed in the following tables. Each column of these tables lists resources for one model, including the url link to the pre-trained model, the dataset that the model is trained on and the total training steps, and several synthesized audio samples based on the pre-trained model.

- Vocoders

<table>
    <thead>
        <tr>
            <th> WaveFlow</th>
            <th><a href="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_ckpt_1.0.zip">ClariNet</a> </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>LJSpeech, 2M</th>
            <th>LJSpeech, 500K</th>
        </tr>
        <tr>
            <th>
            To be added soon
            </th>
            <th>
            <audio id="audio" controls="" preload="none">
      <source id="mp3" src="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_samples_1.0/step_500000_sentence_0.wav">
           </audio> <br>
            <audio id="audio" controls="" preload="none">
      <source id="mp3" src="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_samples_1.0/step_500000_sentence_1.wav">
      </audio><br>
            <audio id="audio" controls="" preload="none">
      <source id="mp3" src="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_samples_1.0/step_500000_sentence_2.wav">
      </audio><br>
      <audio id="audio" controls="" preload="none">
      <source id="mp3" src="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_samples_1.0/step_500000_sentence_3.wav">
      </audio><br>
      <audio id="audio" controls="" preload="none">
      <source id="mp3" src="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_samples_1.0/step_500000_sentence_4.wav">
      </audio>
            </th>

        </tr>
    </tbody>
</table>

**Note:** The input samples are drawn from validation dataset that are not visible in training.

- TTS models

Click each link to download, then one can get the compressed package which contains the pre-trained model and the `yaml` config describing how to train the model.


## Copyright and License

Parakeet is provided under the [Apache-2.0 license](LICENSE).
