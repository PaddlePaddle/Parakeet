
# train model
CUDA_VISIBLE_DEVICES=0 \
python -u synthesis.py \
--max_len=400 \
--use_gpu=0 \
--output='./synthesis' \
--config='configs/ljspeech.yaml' \
--checkpoint_transformer='./checkpoint/transformer/step-120000' \
--vocoder='wavenet' \
--config_vocoder='../wavenet/config.yaml' \
--checkpoint_vocoder='../wavenet/step-2450000' \
#--vocoder='waveflow' \
#--config_vocoder='../waveflow/checkpoint/waveflow_res64_ljspeech_ckpt_1.0/waveflow_ljspeech.yaml' \
#--checkpoint_vocoder='../waveflow/checkpoint/waveflow_res64_ljspeech_ckpt_1.0/step-3020000' \
#--vocoder='cbhg' \
#--config_vocoder='configs/ljspeech.yaml' \
#--checkpoint_vocoder='checkpoint/cbhg/step-100000' \

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi
exit 0
