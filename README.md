# Parakeet

Parakeet aims to provide a flexible, efficient and state-of-the-art text-to-speech toolkit for the open-source community. It is built on Paddle Fluid dynamic graph, with the support of many influential TTS models proposed by [Baidu Research](http://research.baidu.com) and other academic institutions.  

<div align="center">
  <img src="images/logo.png" width=450 /> <br>
</div>

## Installation

### Install Paddlepaddle

See [install](https://www.paddlepaddle.org.cn/install/quick) for more details. This repo requires paddlepaddle's version is above 1.7.

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
nltk.download("cmudict")
```
## Supported models

- [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](./parakeet/models/deepvoice3)

## Examples

- [Train a deepvoice 3 model with ljspeech dataset](./parakeet/examples/deepvoice3) 
