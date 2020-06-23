# train model

CUDA_VISIBLE_DEVICES=0 \
python -u synthesis.py \
--use_gpu=1 \
--alpha=1.0 \
--checkpoint='./checkpoint/fastspeech/step-162000' \
--config='configs/ljspeech.yaml' \
--output='./synthesis' \
--vocoder='waveflow' \
--config_vocoder='../waveflow/checkpoint/waveflow_res128_ljspeech_ckpt_1.0/waveflow_ljspeech.yaml' \
--checkpoint_vocoder='../waveflow/checkpoint/waveflow_res128_ljspeech_ckpt_1.0/step-2000000' \



if [ $? -ne 0 ]; then
    echo "Failed in synthesis!"
    exit 1
fi
exit 0