# Deep Voice 3

PaddlePaddle dynamic graph implementation of Deep Voice 3, a convolutional network based text-to-speech generative model. The implementation is based on [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654).

We implement Deep Voice 3 using Paddle Fluid with dynamic graph, which is convenient for building flexible network architectures.

## Dataset

We experiment with the LJSpeech dataset. Download and unzip [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).

```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjvf LJSpeech-1.1.tar.bz2
```

## Model Architecture

![Deep Voice 3 model architecture](./images/model_architecture.png)

The model consists of an encoder, a decoder and a converter (and a speaker embedding for multispeaker models). The encoder and the decoder together form the seq2seq part of the model, and the converter forms the postnet part.

## Project Structure

```text
├── data.py          data_processing
├── ljspeech.yaml    (example) configuration file
├── sentences.txt    sample sentences
├── synthesis.py     script to synthesize waveform from text
├── train.py         script to train a model
└── utils.py         utility functions
```

## Train

Train the model using train.py, follow the usage displayed by `python train.py --help`.

```text
usage: train.py [-h] [-c CONFIG] [-s DATA] [-r RESUME] [-o OUTPUT] [-g DEVICE]

Train a Deep Voice 3 model with LJSpeech dataset.

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

- `--config` is the configuration file to use. The provided `ljspeech.yaml` can be used directly. And you can change some values in the configuration file and train the model with a different config.
- `--data` is the path of the LJSpeech dataset, the extracted folder from the downloaded archive (the folder which contains metadata.txt).
- `--resume` is the path of the checkpoint. If it is provided, the model would load the checkpoint before trainig.
- `--output` is the directory to save results, all results are saved in this directory. The structure of the output directory is shown below.

```text
├── checkpoints      # checkpoint
├── log              # tensorboard log
└── states           # train and evaluation results
    ├── alignments   # attention
    ├── lin_spec     # linear spectrogram
    ├── mel_spec     # mel spectrogram
    └── waveform     # waveform (.wav files)
```

- `--device` is the device (gpu id) to use for training. `-1` means CPU.

Example script:

```bash
python train.py --config=./ljspeech.yaml --data=./LJSpeech-1.1/ --output=experiment --device=0
```

You can monitor training log via tensorboard, using the script below.

```bash
cd experiment/log
tensorboard --logdir=.
```

## Synthesis
```text
usage: synthesis.py [-h] [-c CONFIG] [-g DEVICE] checkpoint text output_path

Synthsize waveform from a checkpoint.

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

- `--config` is the configuration file to use. You should use the same configuration with which you train you model.
- `checkpoint` is the checkpoint to load.
- `text`is the text file to synthesize.
- `output_path` is the directory to save results. The output path contains the generated audio files (`*.wav`) and attention plots (*.png) for each sentence.
- `--device` is the device (gpu id) to use for training. `-1` means CPU.

Example script:

```bash
python synthesis.py --config=./ljspeech.yaml --device=0 experiment/checkpoints/model_step_005000000 sentences.txt generated
```
