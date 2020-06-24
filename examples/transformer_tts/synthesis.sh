
# train model
CUDA_VISIBLE_DEVICES=0 \
python -u synthesis.py \
--use_gpu=0 \
--output='./synthesis' \
--config='transformer_tts_ljspeech_ckpt_1.0/ljspeech.yaml' \
--checkpoint_transformer='./transformer_tts_ljspeech_ckpt_1.0/step-120000' \
--vocoder='waveflow' \
--config_vocoder='./waveflow_res128_ljspeech_ckpt_1.0/waveflow_ljspeech.yaml' \
--checkpoint_vocoder='./waveflow_res128_ljspeech_ckpt_1.0/step-2000000' \

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi
exit 0
