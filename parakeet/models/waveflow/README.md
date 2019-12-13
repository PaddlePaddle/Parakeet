### Install

pip install -r requirements.txt

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

### Train on multiple GPUs

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/../../.."
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u -m paddle.distributed.launch train.py \
    --config=./configs/waveflow_ljspeech_sqz16_r64_layer8x8.yaml \
    --root=./data/LJSpeech-1.1 \
    --name=test_speed --parallel=true --use_gpu=true
```

Use `export CUDA_VISIBLE_DEVICES=0,1,2,3` to set the GPUs that you want to use to be visible. Then the `paddle.distributed.launch` module will use these visible GPUs to do data parallel training in multiprocessing mode.
