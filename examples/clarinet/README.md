# Clarinet

Paddle implementation of clarinet in dynamic graph, a convolutional network based vocoder. The implementation is based on the paper [ClariNet: Parallel Wave Generation in End-to-End Text-to-Speech](arxiv.org/abs/1807.07281).


## Dataset

We experiment with the LJSpeech dataset. Download and unzip [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).

```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjvf LJSpeech-1.1.tar.bz2
```

## Project Structure

```text
├── data.py          data_processing
├── configs/         (example) configuration file
├── synthesis.py     script to synthesize waveform from mel_spectrogram
├── train.py         script to train a model
└── utils.py         utility functions
```

## Train

Train the model using train.py, follow the usage displayed by `python train.py --help`.

```text
usage: train.py [-h] [--config CONFIG] [--device DEVICE] [--output OUTPUT]
                [--data DATA] [--resume RESUME] [--conditioner CONDITIONER]
                [--teacher TEACHER]

train a clarinet model with LJspeech and a trained wavenet model.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       path of the config file.
  --device DEVICE       device to use.
  --output OUTPUT       path to save student.
  --data DATA           path of LJspeech dataset.
  --resume RESUME       checkpoint to load from.
  --conditioner CONDITIONER
                        conditioner checkpoint to use.
  --teacher TEACHER     teacher checkpoint to use.
```

1. `--config` is the configuration file to use. The provided configurations can be used directly. And you can change some values in the configuration file and train the model with a different config.
2. `--data` is the path of the LJSpeech dataset, the extracted folder from the downloaded archive (the folder which contains metadata.txt).
3. `--resume` is the path of the checkpoint. If it is provided, the model would load the checkpoint before trainig.
4. `--output` is the directory to save results, all result are saved in this directory. The structure of the output directory is shown below.

```text
├── checkpoints      # checkpoint
├── states           # audio files generated at validation
└── log              # tensorboard log
```

5. `--device` is the device (gpu id) to use for training. `-1` means CPU.
6. `--conditioner` is the path of the checkpoint to load for the `conditioner` part of clarinet. if you do not specify `--resume`, then this must be provided.
7. `--teacher` is the path of the checkpoint to load for the `teacher` part of clarinet. if you do not specify `--resume`, then this must be provided.

Before you start training a clarinet model, you should have trained a wavenet model with single gaussian as output distribution. Make sure the config for teacher matches that for the trained model.

example script:

```bash
python train.py --config=./configs/clarinet_ljspeech.yaml --data=./LJSpeech-1.1/ --output=experiment --device=0 --conditioner=wavenet_checkpoint/conditioner --conditioner=wavenet_checkpoint/teacher
```

You can monitor training log via tensorboard, using the script below.

```bash
cd experiment/log
tensorboard --logdir=.
```

## Synthesis
```text
usage: synthesis.py [-h] [--config CONFIG] [--device DEVICE] [--data DATA]
                    checkpoint output

train a clarinet model with LJspeech and a trained wavenet model.

positional arguments:
  checkpoint       checkpoint to load from.
  output           path to save student.

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  path of the config file.
  --device DEVICE  device to use.
  --data DATA      path of LJspeech dataset.
```

1. `--config` is the configuration file to use. You should use the same configuration with which you train you model.
2.  `--data` is the path of the LJspeech dataset. A dataset is not needed for synthesis, but since the input is mel spectrogram, we need to get mel spectrogram from audio files.
3. `checkpoint` is the checkpoint to load.
4. `output_path` is the directory to save results. The output path contains the generated audio files (`*.wav`).
5. `--device` is the device (gpu id) to use for training. `-1` means CPU.

example script:

```bash
python synthesis.py --config=./configs/wavenet_single_gaussian.yaml --data=./LJSpeech-1.1/ --device=0 experiment/checkpoints/step_500000 generated
```
