# Speedyspeech with the Baker dataset

This example contains code used to train a [Speedyspeech](http://arxiv.org/abs/2008.03802) model with [Chinese Standard Mandarin Speech Copus](https://www.data-baker.com/open_source.html). NOTE that we only implement the student part of the Speedyspeech model. The ground truth alignment used to train the model is extracted from the dataset.

## Preprocess the dataset

Download the dataset from the [official website of data-baker](https://www.data-baker.com/data/index/source) and extract it to `~/datasets`. Then the dataset is in directory `~/datasets/BZNSYP`.

Run the script for preprocessing.

```bash
bash preprocess.sh
```

When it is done. A `dump` folder is created in the current directory. The structure of the dump folder is listed below.

```text
dump
├── dev
│   ├── norm
│   └── raw
├── test
│   ├── norm
│   └── raw
└── train
    ├── norm
    ├── raw
    └── stats.npy
```

The dataset is split into 3 parts, namely `train`, `dev` and `test`, each of which contains a `norm` and `raw` sub folder. The raw folder contains log magnitude of mel spectrogram of each utterances, while the norm folder contains normalized spectrogram. The statistics used to normalize the spectrogram is computed from the training set, which is located in `dump/train/stats.npy`.

Also there is a `metadata.jsonl` in each subfolder. It is a table-like file which contains phones, tones, durations, path of spectrogram, and id of each utterance.

## Train the model

To train the model use the `run.sh`. It is an example script to run `train.py`.

```bash
bash run.sh
```

Or you can use `train.py` directly. Here's the complete help message.

```text
usage: train.py [-h] [--config CONFIG] [--train-metadata TRAIN_METADATA]
                [--dev-metadata DEV_METADATA] [--output-dir OUTPUT_DIR]
                [--device DEVICE] [--nprocs NPROCS] [--verbose VERBOSE]

Train a Speedyspeech model with Baker Mandrin TTS dataset.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       config file to overwrite default config
  --train-metadata TRAIN_METADATA
                        training data
  --dev-metadata DEV_METADATA
                        dev data
  --output-dir OUTPUT_DIR
                        output dir
  --device DEVICE       device type to use
  --nprocs NPROCS       number of processes
  --verbose VERBOSE     verbose
```

1. `--config` is a config file in yaml format to overwrite the default config, which can be found at `conf/default.yaml`.
2. `--train-metadata` and `--dev-metadata` should be the metadata file in the normalized subfolder of `train` and `dev` in the `dump` folder.
3. `--output-dir` is the directory to save the results of the experiment. Checkpoints are save in `checkpoints/` inside this directory.
4. `--device` is the type of the device to run the experiment, 'cpu' or 'gpu' are supported.
5. `--nprocs` is the number of processes to run in parallel, note that nprocs > 1 is only supported when `--device` is 'gpu'.

## Pretrained Models

Pretrained models can be downloaded here:
1. Speedyspeech checkpoint. [speedyspeech_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/speedyspeech_baker_ckpt_0.4.zip)
2. Parallel WaveGAN checkpoint. [pwg_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/pwg_baker_ckpt_0.4.zip), which is used as a vocoder in the end-to-end inference script.

Speedyspeech checkpoint contains files listed below.

```text
speedyspeech_baker_ckpt_0.4
├── speedyspeech_default.yaml             # default config used to train speedyseech
├── speedy_speech_stats.npy               # statistics used to normalize spectrogram when training speedyspeech
└── speedyspeech_snapshot_iter_91800.pdz  # model parameters and optimizer states
```

Parallel WaveGAN checkpoint contains files listed below.

```text
pwg_baker_ckpt_0.4
├── pwg_default.yaml              # default config used to train parallel wavegan
├── pwg_snapshot_iter_400000.pdz  # model parameters and optimizer states of parallel wavegan
└── pwg_stats.npy                 # statistics used to normalize spectrogram when training parallel wavegan
```

## Synthesize End to End

When training is done or pretrained models are downloaded. You can run `synthesize_e2e.py` to synthsize.

```text
usage: synthesize_e2e.py [-h] [--speedyspeech-config SPEEDYSPEECH_CONFIG]
                         [--speedyspeech-checkpoint SPEEDYSPEECH_CHECKPOINT]
                         [--speedyspeech-stat SPEEDYSPEECH_STAT]
                         [--pwg-config PWG_CONFIG]
                         [--pwg-checkpoint PWG_CHECKPOINT]
                         [--pwg-stat PWG_STAT] [--text TEXT]
                         [--phones-dict PHONES_DICT] [--tones-dict TONES_DICT]
                         [--output-dir OUTPUT_DIR]
                         [--inference-dir INFERENCE_DIR] [--device DEVICE]
                         [--verbose VERBOSE]

Synthesize with speedyspeech & parallel wavegan.

optional arguments:
  -h, --help            show this help message and exit
  --speedyspeech-config SPEEDYSPEECH_CONFIG
                        config file for speedyspeech.
  --speedyspeech-checkpoint SPEEDYSPEECH_CHECKPOINT
                        speedyspeech checkpoint to load.
  --speedyspeech-stat SPEEDYSPEECH_STAT
                        mean and standard deviation used to normalize
                        spectrogram when training speedyspeech.
  --pwg-config PWG_CONFIG
                        config file for parallelwavegan.
  --pwg-checkpoint PWG_CHECKPOINT
                        parallel wavegan checkpoint to load.
  --pwg-stat PWG_STAT   mean and standard deviation used to normalize
                        spectrogram when training speedyspeech.
  --text TEXT           text to synthesize, a 'utt_id sentence' pair per line
  --phones-dict PHONES_DICT
                        phone vocabulary file.
  --tones-dict TONES_DICT
                        tone vocabulary file.
  --output-dir OUTPUT_DIR
                        output dir
  --inference-dir INFERENCE_DIR
                        dir to save inference models
  --device DEVICE       device type to use
  --verbose VERBOSE     verbose
```

1. `--speedyspeech-config`, `--speedyspeech-checkpoint`, `--speedyspeech-stat` are arguments for speedyspeech, which correspond to the 3 files in the speedyspeech pretrained model.
2. `--pwg-config`, `--pwg-checkpoint`, `--pwg-stat` are arguments for parallel wavegan, which correspond to the 3 files in the parallel wavegan pretrained model.
3. `--text` is the text file, which contains sentences to synthesize.
4. `--output-dir` is the directory to save synthesized audio files.
5. `--inference-dir` is the directory to save exported model, which can be used with paddle infernece.
6. `--device` is the type of device to run synthesis, 'cpu' and 'gpu' are supported. 'gpu' is recommended for faster synthesis.
6. `--phones-dict` is the path of the phone vocabulary file.
7. `--tones-dict` is the path of the tone vocabulary file.
