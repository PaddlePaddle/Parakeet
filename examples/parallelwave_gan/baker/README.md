# Parallel WaveGAN with the Baker dataset

This example contains code used to train a parallel wavegan model with [Chinese Standard Mandarin Speech Copus](https://www.data-baker.com/open_source.html).

## Preprocess the dataset

Download the dataset from the [official website of data-baker](https://www.data-baker.com/data/index/source) and extract it to `~/datasets`. Then the dataset is `~/datasets/BZNSYP`.

Run the script for preprocessing.

```bash
bash preprocess.sh
```

When it is done. A `dump` folder is created in the current directory. The structure of the dump folder is listed below.

```text
dump
├── dev
│   ├── norm
│   └── raw
├── test
│   ├── norm
│   └── raw
└── train
    ├── norm
    ├── raw
    └── stats.npy
```

The dataset is split into 3 parts, namely train, dev and test, each of which contains a `norm` and `raw` sub folder. The raw folder contains log magnitude of mel spectrogram of each utterances, while the norm folder contains normalized spectrogram. The statistics used to normalize the spectrogram is computed from the training set, which is located in `dump/train/stats.npy`.

Also there is a `metadata.jsonl` in each subfolder. It is a table-like file which contains ids and paths to spectrogam of each utterance.

## Train the model

To train the model use the `run.sh`.

```bash
bash run.sh
```

Or you can use the `train.py` script. Here's the complete help message to run it.

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
3. `--output-dir` is the directory to save the results of the experiment.
4. `--device` is the type of the device to run the experiment, 'cpu' or 'gpu' are supported.
5. `--nprocs` is the number of processes to run in parallel, note that nprocs > 1 is only supported when `--device` is 'gpu'.

## Pretrained Models

Pretrained models can be downloaded here:
1. Parallel WaveGAN checkpoint. [pwg_baker_ckpt_0.4.zip](https://paddlespeech.bj.bcebos.com/Parakeet/pwg_baker_ckpt_0.4.zip), which is used as a vocoder in the end-to-end inference script.

Parallel WaveGAN checkpoint contains files listed below.

```text
pwg_baker_ckpt_0.4
├── pwg_default.yaml              # default config used to train parallel wavegan
├── pwg_snapshot_iter_400000.pdz  # generator parameters of parallel wavegan
└── pwg_stats.npy                 # statistics used to normalize spectrogram when training parallel wavegan
```

## Synthesize

When training is done or pretrained models are downloaded. You can run `synthesize.py` to synthsize.

```text
usage: synthesize.py [-h] [--config CONFIG] [--checkpoint CHECKPOINT]
                     [--test-metadata TEST_METADATA] [--output-dir OUTPUT_DIR]
                     [--device DEVICE] [--verbose VERBOSE]

synthesize with parallel wavegan.

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       config file to overwrite default config
  --checkpoint CHECKPOINT
                        snapshot to load
  --test-metadata TEST_METADATA
                        dev data
  --output-dir OUTPUT_DIR
                        output dir
  --device DEVICE       device to run
  --verbose VERBOSE     verbose
```

1. `--config` is the extra configuration file to overwrite the default config. You should use the same config with which the model is trained.
2. `--checkpoint` is the checkpoint to load. Pick one of the checkpoints from `/checkpoints` inside the training output directory. If you use the pretrained model, use the `pwg_snapshot_iter_400000.pdz`.
3. `--test-metadata` is the metadata of the test dataset. Use the `metadata.jsonl` in the `dev/norm` subfolder from the processed directory.
4. `--output-dir` is the directory to save the synthesized audio files.
5. `--device` is the type of device to run synthesis, 'cpu' and 'gpu' are supported.
