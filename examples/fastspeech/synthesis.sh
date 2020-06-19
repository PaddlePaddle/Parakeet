# train model

CUDA_VISIBLE_DEVICES=0 \
python -u synthesis.py \
--use_gpu=1 \
--alpha=1.0 \
--checkpoint='./checkpoint/fastspeech1024/step-160000' \
--config='configs/ljspeech.yaml' \
--output='./synthesis' \
--vocoder='waveflow' \
--config_vocoder='../waveflow/checkpoint/waveflow_res64_ljspeech_ckpt_1.0/waveflow_ljspeech.yaml' \
--checkpoint_vocoder='../waveflow/checkpoint/waveflow_res64_ljspeech_ckpt_1.0/step-3020000' \
#--vocoder='clarinet' \
#--config_vocoder='../clarinet/configs/clarinet_ljspeech.yaml' \
#--checkpoint_vocoder='../clarinet/checkpoint/step-500000' \



if [ $? -ne 0 ]; then
    echo "Failed in synthesis!"
    exit 1
fi
exit 0