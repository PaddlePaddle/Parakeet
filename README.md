# Parakeet

Parakeet aims to provide a flexible, efficient and state-of-the-art text-to-speech toolkit for the open-source community. It is built on PaddlePaddle Fluid dynamic graph and includes many influential TTS models proposed by [Baidu Research](http://research.baidu.com) and other research groups.  

In particular, it features the latest [WaveFlow] (https://arxiv.org/abs/1912.01219) model proposed by Baidu Research. 
- WaveFlow can synthesize 22.05 kHz high-fidelity speech around 40x faster than real-time on a Nvidia V100 GPU without engineered inference kernels, which is faster than WaveGlow and serveral orders of magnitude faster than WaveNet.
- WaveFlow is a small-footprint flow-based model for raw audio. It has only 5.9M parameters, which is 15x smalller than WaveGlow (87.9M) and comparable to WaveNet (4.6M).
- WaveFlow is directly trained with maximum likelihood without probability density distillation and auxiliary losses as used in Parallel WaveNet and ClariNet, which simplifies the training pipeline and reduces the cost of development. 


<div align="center">
  <img src="images/logo.png" width=450 /> <br>
</div>

### Setup

Make sure the library `libsndfile1` is installed, e.g., on Ubuntu.

```bash
sudo apt-get install libsndfile1
```

### Install PaddlePaddle

See [install](https://www.paddlepaddle.org.cn/install/quick) for more details. This repo requires paddlepaddle 1.7 or above.

### Install Parakeet

```bash
# git clone this repo first
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


## Related Research

- [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654)
- [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895)
- [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263).
- [WaveFlow: A Compact Flow-based Model for Raw Audio](https://arxiv.org/abs/1912.01219)

## Examples

- [Train a DeepVoice3 model with ljspeech dataset](./examples/deepvoice3)
- [Train a TransformerTTS  model with ljspeech dataset](./examples/transformer_tts)
- [Train a FastSpeech model with ljspeech dataset](./examples/fastspeech)
- [Train a WaveFlow model with ljspeech dataset](./examples/waveflow)

## Copyright and License

Parakeet is provided under the [Apache-2.0 license](LICENSE).
