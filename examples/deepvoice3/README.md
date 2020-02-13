# Deepvoice 3 

Paddle implementation of deepvoice 3 in dynamic graph, a convolutional network based text-to-speech synthesis model. The implementation is based on [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654).

We implement Deepvoice 3 in paddle fluid with dynamic graph, which is convenient for flexible network architectures.

## Installation

### Install paddlepaddle. 
This implementation requires the latest develop version of paddlepaddle. You can either download the compiled package or build paddle from source.

1. Install the compiled package, via pip, conda or docker. See [**Installation Mannuals**](https://www.paddlepaddle.org.cn/documentation/docs/en/beginners_guide/install/index_en.html) for more details.

2. Build paddlepaddle from source. See [**Compile From Source Code**](https://www.paddlepaddle.org.cn/documentation/docs/en/beginners_guide/install/compile/fromsource_en.html) for more details. Note that if you want to enable data parallel training for multiple GPUs, you should set `-DWITH_DISTRIBUTE=ON` with cmake.

### Install parakeet
You can choose to install via pypi or clone the repository and install manually.

1. Install via pypi.
   ```bash
   pip install parakeet
   ```

2. Install manually.
   ```bash
   git clone <url>
   cd Parakeet/
   pip install -e .
   ```

### cmudict
You also need to download cmudict for nltk, because convert text into phonemes with `cmudict`.

```python
import nltk
nltk.download("punkt")
nltk.download("cmudict")
```

## dataset

We experiment with the LJSpeech dataset. Download and unzip [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).

```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjvf LJSpeech-1.1.tar.bz2
```

## Model Architecture

![DeepVoice3 model architecture](./_images/model_architecture.png)

The model consists of an encoder, a decoder and a converter (and a speaker embedding for multispeaker models). The encoder, together with the decoder forms the seq2seq part of the model, and the converter forms the postnet part.

## Project Structure

├── data.py          data_processing 
├── ljspeech.yaml    (example) configuration file
├── sentences.txt    sample sentences
├── synthesis.py     script to synthesize waveform from text
├── train.py         script to train a model
└── utils.py         utility functions

## train

Train the model using train.py, follow the usage displayed by `python train.py --help`.

```text
usage: train.py [-h] [-c CONFIG] [-s DATA] [-r RESUME] [-o OUTPUT] [-g DEVICE]

Train a deepvoice 3 model with LJSpeech dataset.

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        experimrnt config
  -s DATA, --data DATA  The path of the LJSpeech dataset.
  -r RESUME, --resume RESUME
                        checkpoint to load
  -o OUTPUT, --output OUTPUT
                        The directory to save result.
  -g DEVICE, --device DEVICE
                        device to use
``` 

1. `--config` is the configuration file to use. The provided `ljspeech.yaml` can be used directly. And you can change some values in the configuration file and train the model with a different config.
2. `--data` is the path of the LJSpeech dataset, the extracted folder from the downloaded archive (the folder which contains metadata.txt).
3. `--resume` is the path of the checkpoint. If it is provided, the model would load the checkpoint before trainig.
4. `--output` is the directory to save results, all result are saved in this directory. The structure of the output directory is shown below.

```text
├── checkpoints      # checkpoint
├── log              # tensorboard log
└── states           # train and evaluation results
    ├── alignments   # attention 
    ├── lin_spec     # linear spectrogram
    ├── mel_spec     # mel spectrogram
    └── waveform     # waveform (.wav files)
```

5. `--device` is the device (gpu id) to use for training. `-1` means CPU.

## synthesis
```text
usage: synthesis.py [-h] [-c CONFIG] [-g DEVICE] checkpoint text output_path

Synthsize waveform with a checkpoint.

positional arguments:
  checkpoint            checkpoint to load.
  text                  text file to synthesize
  output_path           path to save results

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        experiment config.
  -g DEVICE, --device DEVICE
                        device to use
```

1. `--config` is the configuration file to use. You should use the same configuration with which you train you model.
2. `checkpoint` is the checkpoint to load.
3. `text`is the text file to synthesize.
4. `output_path` is the directory to save results. The output path contains the generated audio files (`*.wav`) and attention plots (*.png) for each sentence.
5. `--device` is the device (gpu id) to use for training. `-1` means CPU.

