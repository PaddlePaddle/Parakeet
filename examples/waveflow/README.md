# WaveFlow

PaddlePaddle dynamic graph implementation of [WaveFlow: A Compact Flow-based Model for Raw Audio](https://arxiv.org/abs/1912.01219).

- WaveFlow can synthesize 22.05 kHz high-fidelity speech around 40x faster than real-time on a Nvidia V100 GPU without engineered inference kernels, which is faster than [WaveGlow](https://github.com/NVIDIA/waveglow) and serveral orders of magnitude faster than WaveNet.
- WaveFlow is a small-footprint flow-based model for raw audio. It has only 5.9M parameters, which is 15x smalller than WaveGlow (87.9M).
- WaveFlow is directly trained with maximum likelihood without probability density distillation and auxiliary losses as used in Parallel WaveNet and ClariNet, which simplifies the training pipeline and reduces the cost of development.

## Project Structure
```text
├── configs                                          # yaml configuration files of preset model hyperparameters
├── benchmark.py                                     # benchmark code to test the speed of batched speech synthesis
├── synthesis.py                                     # script for speech synthesis
├── train.py                                         # script for model training
├── utils.py                                         # helper functions for e.g., model checkpointing
├── data.py                                          # dataset and dataloader settings for LJSpeech
├── waveflow.py                                      # WaveFlow model high level APIs
└── parakeet/models/waveflow/waveflow_modules.py     # WaveFlow model implementation
```

## Usage

There are many hyperparameters to be tuned depending on the specification of model and dataset you are working on.
We provide `wavenet_ljspeech.yaml` as a hyperparameter set that works well on the LJSpeech dataset.
Note that we use [convolutional queue](https://arxiv.org/abs/1611.09482) at audio synthesis to cache the intermediate hidden states, which will speed up the autoregressive inference over the height dimension. Current implementation only supports height dimension equals 8 or 16, i.e., where there is no dilation on the height dimension. Therefore, you can only set value of `n_group` key in the yaml config file to be either 8 or 16.

Also note that `train.py`, `synthesis.py`, and `benchmark.py` all accept a `--config` parameter. To ensure consistency, you should use the same config yaml file for both training, synthesizing and benchmarking. You can also overwrite these preset hyperparameters with command line by updating parameters after `--config`.
For example `--config=${yaml} --batch_size=8` can overwrite the corresponding hyperparameters in the `${yaml}` config file. For more details about these hyperparameters, check `utils.add_config_options_to_parser`.

Additionally, you need to specify some additional parameters for `train.py`, `synthesis.py`, and `benchmark.py`, and the details can be found in `train.add_options_to_parser`, `synthesis.add_options_to_parser`, and `benchmark.add_options_to_parser`, respectively.

### Dataset

Download and unzip [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).

```bash
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar xjvf LJSpeech-1.1.tar.bz2
```

In this example, assume that the path of unzipped LJSpeech dataset is `./data/LJSpeech-1.1`.

### Train on single GPU

```bash
export CUDA_VISIBLE_DEVICES=0
python -u train.py \
    --config=./configs/waveflow_ljspeech.yaml \
    --root=./data/LJSpeech-1.1 \
    --name=${ModelName} --batch_size=4 \
    --use_gpu=true
```

#### Save and Load checkpoints

Our model will save model parameters as checkpoints in `./runs/waveflow/${ModelName}/checkpoint/` every 10000 iterations by default, where `${ModelName}` is the model name for one single experiment and it could be whatever you like.
The saved checkpoint will have the format of `step-${iteration_number}.pdparams` for model parameters and `step-${iteration_number}.pdopt` for optimizer parameters.

There are three ways to load a checkpoint and resume training (take an example that you want to load a 500000-iteration checkpoint):
1. Use `--checkpoint=./runs/waveflow/${ModelName}/checkpoint/step-500000` to provide a specific path to load. Note that you only need to provide the base name of the parameter file, which is `step-500000`, no extension name `.pdparams` or `.pdopt` is needed.
2. Use `--iteration=500000`.
3. If you don't specify either `--checkpoint` or `--iteration`, the model will automatically load the latest checkpoint in `./runs/waveflow/${ModelName}/checkpoint`.

### Train on multiple GPUs

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u -m paddle.distributed.launch train.py \
    --config=./configs/waveflow_ljspeech.yaml \
    --root=./data/LJSpeech-1.1 \
    --name=${ModelName} --use_gpu=true
```

Use `export CUDA_VISIBLE_DEVICES=0,1,2,3` to set the GPUs that you want to use to be visible. Then the `paddle.distributed.launch` module will use these visible GPUs to do data parallel training in multiprocessing mode.

### Monitor with Tensorboard

By default, the logs are saved in `./runs/waveflow/${ModelName}/logs/`. You can monitor logs using TensorBoard.

```bash
tensorboard --logdir=${log_dir} --port=8888
```

### Synthesize from a checkpoint

Check the [Save and load checkpoint](#save-and-load-checkpoints) section on how to load a specific checkpoint.
The following example will automatically load the latest checkpoint:

```bash
export CUDA_VISIBLE_DEVICES=0
python -u synthesis.py \
    --config=./configs/waveflow_ljspeech.yaml \
    --root=./data/LJSpeech-1.1 \
    --name=${ModelName} --use_gpu=true \
    --output=./syn_audios \
    --sample=${SAMPLE} \
    --sigma=1.0
```

In this example, `--output` specifies where to save the synthesized audios and `--sample` (<16) specifies which sample in the valid dataset (a split from the whole LJSpeech dataset, by default contains the first 16 audio samples) to synthesize based on the mel-spectrograms computed from the ground truth sample audio, e.g., `--sample=0` means to synthesize the first audio in the valid dataset.

### Benchmarking

Use the following example to benchmark the speed of batched speech synthesis, which reports how many times faster than real-time:

```bash
export CUDA_VISIBLE_DEVICES=0
python -u benchmark.py \
    --config=./configs/waveflow_ljspeech.yaml \
    --root=./data/LJSpeech-1.1 \
    --name=${ModelName} --use_gpu=true
```

### Low-precision inference

This model supports the float16 low-precision inference. By appending the argument

```bash
    --use_fp16=true
```

to the command of synthesis and benchmarking, one can experience the fast speed of low-precision inference.
