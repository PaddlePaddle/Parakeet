# WaveNet

PaddlePaddle dynamic graph implementation of WaveNet, a convolutional network based vocoder. WaveNet is originally proposed in [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499). However, in this experiment, the implementation follows the teacher model in [ClariNet: Parallel Wave Generation in End-to-End Text-to-Speech](arxiv.org/abs/1807.07281).


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

## Saving & Loading
`train.py` and `synthesis.py` have 3 arguments in common, `--checkpooint`, `iteration` and `output`.

1. `output` is the directory for saving results.
During training, checkpoints are saved in `checkpoints/` in `output` and tensorboard log is save in `log/` in `output`. Other possible outputs are saved in `states/` in `outuput`.
During synthesizing, audio files and other possible outputs are save in `synthesis/` in `output`.
So after training and synthesizing with the same output directory, the file structure of the output directory looks like this.

```text
├── checkpoints/      # checkpoint directory (including *.pdparams, *.pdopt and a text file `checkpoint` that records the latest checkpoint)
├── states/           # audio files generated at validation and other possible outputs
├── log/              # tensorboard log
└── synthesis/        # synthesized audio files and other possible outputs
```

2. `--checkpoint` and `--iteration` for loading from existing checkpoint. Loading existing checkpoiont follows the following rule:
If `--checkpoint` is provided, the checkpoint specified by `--checkpoint` is loaded.
If `--checkpoint` is not provided, we try to load the model specified by `--iteration` from the checkpoint directory. If `--iteration` is not provided, we try to load the latested checkpoint from checkpoint directory.

## Train

Train the model using train.py. For help on usage, try `python train.py --help`.

```text
usage: train.py [-h] [--data DATA] [--config CONFIG] [--device DEVICE]
                [--checkpoint CHECKPOINT | --iteration ITERATION]
                output

Train a WaveNet model with LJSpeech.

positional arguments:
  output                        path to save results

optional arguments:
  -h, --help                    show this help message and exit
  --data DATA                   path of the LJspeech dataset
  --config CONFIG               path of the config file
  --device DEVICE               device to use
  --checkpoint CHECKPOINT       checkpoint to resume from
  --iteration ITERATION         the iteration of the checkpoint to load from output directory
```

- `--data` is the path of the LJSpeech dataset, the extracted folder from the downloaded archive (the folder which contains metadata.txt).
- `--config` is the configuration file to use. The provided configurations can be used directly. And you can change some values in the configuration file and train the model with a different config.
- `--device` is the device (gpu id) to use for training. `-1` means CPU.

- `--checkpoint` is the path of the checkpoint.
- `--iteration` is the iteration of the checkpoint to load from output directory.
- `output` is the directory to save results, all result are saved in this directory.

See [Saving-&-Loading](#Saving-&-Loading) for details of checkpoint loading.


Example script:

```bash
python train.py \
    --config=./configs/wavenet_single_gaussian.yaml \
    --data=./LJSpeech-1.1/ \
    --device=0 \
    experiment
```

You can monitor training log via TensorBoard, using the script below.

```bash
cd experiment/log
tensorboard --logdir=.
```

## Synthesis
```text
usage: synthesis.py [-h] [--data DATA] [--config CONFIG] [--device DEVICE]
                    [--checkpoint CHECKPOINT | --iteration ITERATION]
                    output

Synthesize valid data from LJspeech with a wavenet model.

positional arguments:
  output                        path to save the synthesized audio

optional arguments:
  -h, --help                    show this help message and exit
  --data DATA                   path of the LJspeech dataset
  --config CONFIG               path of the config file
  --device DEVICE               device to use
  --checkpoint CHECKPOINT       checkpoint to resume from
  --iteration ITERATION         the iteration of the checkpoint to load from output directory
```

- `--data` is the path of the LJspeech dataset. In principle, a dataset is not needed for synthesis, but since the input is mel spectrogram, we need to get mel spectrogram from audio files.
- `--config` is the configuration file to use. You should use the same configuration with which you train you model.
- `--device` is the device (gpu id) to use for training. `-1` means CPU.
- `--checkpoint` is the checkpoint to load.
- `--iteration` is the iteration of the checkpoint to load from output directory.
- `output` is the directory to save synthesized audio. Audio file is saved in `synthesis/` in `output` directory.
See [Saving-&-Loading](#Saving-&-Loading) for details of checkpoint loading.


Example script:

```bash
python synthesis.py \
    --config=./configs/wavenet_single_gaussian.yaml \
    --data=./LJSpeech-1.1/ \
    --device=0 \
    --checkpoint="experiment/checkpoints/step-1000000" \
    experiment
```

or

```bash
python synthesis.py \
    --config=./configs/wavenet_single_gaussian.yaml \
    --data=./LJSpeech-1.1/ \
    --device=0 \
    --iteration=1000000 \
    experiment
```
