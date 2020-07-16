# Parakeet

Parakeet aims to provide a flexible, efficient and state-of-the-art text-to-speech toolkit for the open-source community. It is built on PaddlePaddle Fluid dynamic graph and includes many influential TTS models proposed by [Baidu Research](http://research.baidu.com) and other research groups.  

<div align="center">
  <img src="images/logo.png" width=450 /> <br>
</div>

In particular, it features the latest [WaveFlow](https://arxiv.org/abs/1912.01219) model proposed by Baidu Research.

- WaveFlow can synthesize 22.05 kHz high-fidelity speech around 40x faster than real-time on a Nvidia V100 GPU without engineered inference kernels, which is faster than [WaveGlow](https://github.com/NVIDIA/waveglow) and serveral orders of magnitude faster than WaveNet.
- WaveFlow is a small-footprint flow-based model for raw audio. It has only 5.9M parameters, which is 15x smalller than WaveGlow (87.9M).
- WaveFlow is directly trained with maximum likelihood without probability density distillation and auxiliary losses as used in Parallel WaveNet and ClariNet, which simplifies the training pipeline and reduces the cost of development.

## Overview

In order to facilitate exploiting the existing TTS models directly and developing the new ones, Parakeet selects typical models and provides their reference implementations in PaddlePaddle. Further more, Parakeet abstracts the TTS pipeline and standardizes the procedure of data preprocessing, common modules sharing, model configuration, and the process of training and synthesis. The models supported here include Vocoders and end-to-end TTS models:

- Vocoders
  - [WaveFlow: A Compact Flow-based Model for Raw Audio](https://arxiv.org/abs/1912.01219)
  - [ClariNet: Parallel Wave Generation in End-to-End Text-to-Speech](https://arxiv.org/abs/1807.07281)
  - [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)

- TTS models
  - [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654)
  - [Neural Speech Synthesis with Transformer Network (Transformer TTS)](https://arxiv.org/abs/1809.08895)
  - [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263)

And more will be added in the future.

See the [guide](docs/experiment_guide.md) for details about how to build your own model and experiment in Parakeet.

## Setup

Make sure the library `libsndfile1` is installed, e.g., on Ubuntu.

```bash
sudo apt-get install libsndfile1
```

### Install PaddlePaddle

See [install](https://www.paddlepaddle.org.cn/install/quick) for more details. This repo requires PaddlePaddle **1.8.2** or above.

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

Parakeet also releases some well-trained parameters for the example models, which can be accessed in the following tables. Each column of these tables lists resources for one model, including the url link to the pre-trained model, the dataset that the model is trained on, and synthesized audio samples based on the pre-trained model. Click each model name to download, then you can get the compressed package which contains the pre-trained model and the `yaml` config describing how the model is trained.

#### Vocoders

We provide the model checkpoints of WaveFlow with 64, 96 and 128 residual channels, ClariNet and WaveNet.

<div align="center">
<table>
    <thead>
        <tr>
            <th  style="width: 250px">
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res64_ljspeech_ckpt_1.0.zip">WaveFlow (res. channels 64)</a>
            </th>
            <th  style="width: 250px">
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res96_ljspeech_ckpt_1.0.zip">WaveFlow (res. channels 96)</a>
            </th>
            <th  style="width: 250px">
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res128_ljspeech_ckpt_1.0.zip">WaveFlow (res. channels 128)</a>
            </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>LJSpeech </th>
            <th>LJSpeech </th>
            <th>LJSpeech </th>
        </tr>
        <tr>
            <th>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res64_ljspeech_samples_1.0/step_3020k_sentence_0.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res64_ljspeech_samples_1.0/step_3020k_sentence_1.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res64_ljspeech_samples_1.0/step_3020k_sentence_2.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res64_ljspeech_samples_1.0/step_3020k_sentence_3.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res64_ljspeech_samples_1.0/step_3020k_sentence_4.wav">
            <img src="images/audio_icon.png" width=250 /></a>
            </th>
            <th>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res96_ljspeech_samples_1.0/step_2000k_sentence_0.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res96_ljspeech_samples_1.0/step_2000k_sentence_1.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res96_ljspeech_samples_1.0/step_2000k_sentence_2.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res96_ljspeech_samples_1.0/step_2000k_sentence_3.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res96_ljspeech_samples_1.0/step_2000k_sentence_4.wav">
            <img src="images/audio_icon.png" width=250 /></a>
            </th>
            <th>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_0.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_1.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_2.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_3.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/waveflow_res128_ljspeech_samples_1.0/step_2000k_sentence_4.wav">
            <img src="images/audio_icon.png" width=250 /></a>
            </th>
        </tr>
    </tbody>
    <thead>
        <tr>
            <th  style="width: 250px">
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_ckpt_1.0.zip">ClariNet</a>
            </th>
            <th  style="width: 250px">
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/wavenet_ljspeech_ckpt_1.0.zip">WaveNet</a>
            </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th>LJSpeech </th>
            <th>LJSpeech </th>
        </tr>
        <tr>
            <th>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_samples_1.0/step_500000_sentence_0.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_samples_1.0/step_500000_sentence_1.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_samples_1.0/step_500000_sentence_2.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_samples_1.0/step_500000_sentence_3.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/clarinet_ljspeech_samples_1.0/step_500000_sentence_4.wav">
            <img src="images/audio_icon.png" width=250 /></a>  
            </th>
            <th>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/wavenet_ljspeech_samples_1.0/step_2450k_sentence_0.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/wavenet_ljspeech_samples_1.0/step_2450k_sentence_1.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/wavenet_ljspeech_samples_1.0/step_2450k_sentence_2.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/wavenet_ljspeech_samples_1.0/step_2450k_sentence_3.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/wavenet_ljspeech_samples_1.0/step_2450k_sentence_4.wav">
            <img src="images/audio_icon.png" width=250 /></a>  
            </th>
        </tr>
    </tbody>
</table>
</div>


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Note:** The input mel spectrogams are from validation dataset, which are not seen during training.

#### TTS models

We also provide checkpoints for different end-to-end TTS models, and present the synthesized audio examples for some randomly chosen famous quotes. The corresponding texts are displayed as follows.

||Text | From |
|:-:|:-- | :--: |
0|*Life was like a box of chocolates, you never know what you're gonna get.* | *Forrest Gump* |  
1|*With great power there must come great responsibility.* | *Spider-Man*|
2|*To be or not to be, that’s a question.*|*Hamlet*|
3|*Death is just a part of life, something we're all destined to do.*| *Forrest Gump*|
4|*Don’t argue with the people of strong determination, because they may change the fact!*| *William Shakespeare* |

Users have the option to use different vocoders to convert the linear/mel spectrogam to the raw audio in TTS models. Taking this into account, we are going to release the checkpoints for TTS models adapted to different vocoders, including the [Griffin-Lim](https://ieeexplore.ieee.org/document/1164317) algorithm and some neural vocoders.

##### 1) Griffin-Lim

<div align="center">
<table>
    <thead>
        <tr>
            <th  style="width: 250px">
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_ckpt_1.0.zip">Transformer TTS</a>
            </th>
            <th  style="width: 250px">
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech_ljspeech_ckpt_1.0.zip">FastSpeech</a>
            </th>
                    </tr>
    </thead>
    <tbody>
        <tr>
            <th>LJSpeech </th>
            <th>LJSpeech </th>
        </tr>
        <tr>
            <th >
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_griffin-lim_samples_1.0/step_120000_sentence_0.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_griffin-lim_samples_1.0/step_120000_sentence_1.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_griffin-lim_samples_1.0/step_120000_sentence_2.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_griffin-lim_samples_1.0/step_120000_sentence_3.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/transformer_tts_ljspeech_griffin-lim_samples_1.0/step_120000_sentence_4.wav">
            <img src="images/audio_icon.png" width=250 /></a>
            </th>
            <th >
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech_ljspeech_griffin-lim_samples_1.0/step_162000_sentence_0.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech_ljspeech_griffin-lim_samples_1.0/step_162000_sentence_1.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech_ljspeech_griffin-lim_samples_1.0/step_162000_sentence_2.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech_ljspeech_griffin-lim_samples_1.0/step_162000_sentence_3.wav">
            <img src="images/audio_icon.png" width=250 /></a><br>
            <a href="https://paddlespeech.bj.bcebos.com/Parakeet/fastspeech_ljspeech_griffin-lim_samples_1.0/step_162000_sentence_4.wav">
            <img src="images/audio_icon.png" width=250 /></a>
            </th>
        </tr>
    </tbody>
    <thead>
</table>
</div>

##### 2) Neural vocoders

under preparation

## Copyright and License

Parakeet is provided under the [Apache-2.0 license](LICENSE).
