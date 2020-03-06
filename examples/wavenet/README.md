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

## Train

Train the model using train.py. For help on usage, try `python train.py --help`.

```text
usage: train.py [-h] [--data DATA] [--config CONFIG] [--output OUTPUT]
                [--device DEVICE] [--resume RESUME]

Train a WaveNet model with LJSpeech.

optional arguments:
  -h, --help       show this help message and exit
  --data DATA      path of the LJspeech dataset.
  --config CONFIG  path of the config file.
  --output OUTPUT  path to save results.
  --device DEVICE  device to use.
  --resume RESUME  checkpoint to resume from.
```

- `--config` is the configuration file to use. The provided configurations can be used directly. And you can change some values in the configuration file and train the model with a different config.
- `--data` is the path of the LJSpeech dataset, the extracted folder from the downloaded archive (the folder which contains metadata.txt).
- `--resume` is the path of the checkpoint. If it is provided, the model would load the checkpoint before training.
- `--output` is the directory to save results, all result are saved in this directory. The structure of the output directory is shown below.

```text
├── checkpoints      # checkpoint
└── log              # tensorboard log
```

- `--device` is the device (gpu id) to use for training. `-1` means CPU.

Example script:

```bash
python train.py --config=./configs/wavenet_single_gaussian.yaml --data=./LJSpeech-1.1/ --output=experiment --device=0
```

You can monitor training log via TensorBoard, using the script below.

```bash
cd experiment/log
tensorboard --logdir=.
```

## Synthesis
```text
usage: synthesis.py [-h] [--data DATA] [--config CONFIG] [--device DEVICE]
                    checkpoint output

Synthesize valid data from LJspeech with a WaveNet model.

positional arguments:
  checkpoint       checkpoint to load.
  output           path to save results.

optional arguments:
  -h, --help       show this help message and exit
  --data DATA      path of the LJspeech dataset.
  --config CONFIG  path of the config file.
  --device DEVICE  device to use.
```

- `--config` is the configuration file to use. You should use the same configuration with which you train you model.
- `--data` is the path of the LJspeech dataset. A dataset is not needed for synthesis, but since the input is mel spectrogram, we need to get mel spectrogram from audio files.
- `checkpoint` is the checkpoint to load.
- `output_path` is the directory to save results. The output path contains the generated audio files (`*.wav`).
- `--device` is the device (gpu id) to use for training. `-1` means CPU.

Example script:

```bash
python synthesis.py --config=./configs/wavenet_single_gaussian.yaml --data=./LJSpeech-1.1/ --device=0 experiment/checkpoints/step_500000 generated
```
