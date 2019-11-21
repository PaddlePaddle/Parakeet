export LD_LIBRARY_PATH=/fluid13_workspace/cuda-9.0/lib64/:/fluid13_workspace/cudnnv7.5_cuda9.0/lib64/:$LD_LIBRARY_PATH
#export PYTHONPATH=/dv3_workspace/paddle_for_dv3/build/python/

export PYTHONPATH=/fluid13_workspace/paddle_cherry_pick/build/python/:../

export CUDA_VISIBLE_DEVICES=7
GLOG_v=0 python -u train.py \
    --use-gpu \
    --reset-optimizer \
    --preset=presets/deepvoice3_ljspeech.json \
    --checkpoint-dir=checkpoint_single_1014 \
    --data-root="/fluid13_workspace/dv3_workspace/deepvoice3_pytorch/data/ljspeech/" \
    --hparams="batch_size=16"
