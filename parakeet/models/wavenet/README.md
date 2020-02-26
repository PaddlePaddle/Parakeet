# WaveNet with Paddle Fluid

Paddle fluid implementation of WaveNet, a deep generative model of raw audio waveforms.
WaveNet model is originally proposed in [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499).
Our implementation is based on the WaveNet architecture described in [ClariNet: Parallel Wave Generation in End-to-End Text-to-Speech](https://arxiv.org/abs/1807.07281) and can provide various output distributions, including single Gaussian, mixture of Gaussian, and softmax with linearly quantized channels.

We implement WaveNet model in paddle fluid with dynamic graph, which is convenient for flexible network architectures.

## Project Structure
```text
├── configs                 # yaml configuration files of preset model hyperparameters
├── data.py                 # dataset and dataloader settings for LJSpeech
├── slurm.py                # optional slurm helper functions if you use slurm to train model
├── synthesis.py            # script for speech synthesis
├── train.py                # script for model training
├── utils.py                # helper functions for e.g., model checkpointing
├── wavenet.py              # WaveNet model high level APIs
└── wavenet_modules.py      # WaveNet model implementation
```

## Usage

There are many hyperparameters to be tuned depending on the specification of model and dataset you are working on. Hyperparameters that are known to work good for the LJSpeech dataset are provided as yaml files in `./configs/` folder. Specifically, we provide `wavenet_ljspeech_single_gaussian.yaml`, `wavenet_ljspeech_mix_gaussian.yaml`, and `wavenet_ljspeech_softmax.yaml` config files for WaveNet with single Gaussian, 10-component mixture of Gaussians, and softmax (with 2048 linearly quantized channels) output distributions, respectively.

Note that `train.py` and `synthesis.py` all accept a `--config` parameter. To ensure consistency, you should use the same config yaml file for both training and synthesizing. You can also overwrite these preset hyperparameters with command line by updating parameters after `--config`. For example `--config=${yaml} --batch_size=8 --layers=20` can overwrite the corresponding hyperparameters in the `${yaml}` config file. For more details about these hyperparameters, check `utils.add_config_options_to_parser`.

Note that you also need to specify some additional parameters for `train.py` and `synthesis.py`, and the details can be found in `train.add_options_to_parser` and `synthesis.add_options_to_parser`, respectively.

### Dataset

Download and unzip [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).

```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjvf LJSpeech-1.1.tar.bz2
```

In this example, assume that the path of unzipped LJSpeech dataset is `./data/LJSpeech-1.1`.

### Train on single GPU

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/../../.."
export CUDA_VISIBLE_DEVICES=0
python -u train.py --config=${yaml} \
    --root=./data/LJSpeech-1.1 \
    --name=${ModelName} --batch_size=4 \
    --parallel=false --use_gpu=true
```

#### Save and Load checkpoints

Our model will save model parameters as checkpoints in `./runs/wavenet/${ModelName}/checkpoint/` every 10000 iterations by default.
The saved checkpoint will have the format of `step-${iteration_number}.pdparams` for model parameters and `step-${iteration_number}.pdopt` for optimizer parameters.

There are three ways to load a checkpoint and resume training (take an example that you want to load a 500000-iteration checkpoint):
1. Use `--checkpoint=./runs/wavenet/${ModelName}/checkpoint/step-500000` to provide a specific path to load. Note that you only need to provide the base name of the parameter file, which is `step-500000`, no extension name `.pdparams` or `.pdopt` is needed.
2. Use `--iteration=500000`.
3. If you don't specify either `--checkpoint` or `--iteration`, the model will automatically load the latest checkpoint in `./runs/wavenet/${ModelName}/checkpoint`.

### Train on multiple GPUs

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/../../.."
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u -m paddle.distributed.launch train.py \
    --config=${yaml} \
    --root=./data/LJSpeech-1.1 \
    --name=${ModelName} --parallel=true --use_gpu=true
```

Use `export CUDA_VISIBLE_DEVICES=0,1,2,3` to set the GPUs that you want to use to be visible. Then the `paddle.distributed.launch` module will use these visible GPUs to do data parallel training in multiprocessing mode.

### Monitor with Tensorboard

By default, the logs are saved in `./runs/wavenet/${ModelName}/logs/`. You can monitor logs by tensorboard.

```bash
tensorboard --logdir=${log_dir} --port=8888
```

### Synthesize from a checkpoint

Check the [Save and load checkpoint](#save-and-load-checkpoints) section on how to load a specific checkpoint.
The following example will automatically load the latest checkpoint:

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/../../.."
export CUDA_VISIBLE_DEVICES=0
python -u synthesis.py --config=${yaml} \
    --root=./data/LJSpeech-1.1 \
    --name=${ModelName} --use_gpu=true \
    --output=./syn_audios \
    --sample=${SAMPLE}
```

In this example, `--output` specifies where to save the synthesized audios and `--sample` specifies which sample in the valid dataset (a split from the whole LJSpeech dataset, by default contains the first 16 audio samples) to synthesize based on the mel-spectrograms computed from the ground truth sample audio, e.g., `--sample=0` means to synthesize the first audio in the valid dataset.
