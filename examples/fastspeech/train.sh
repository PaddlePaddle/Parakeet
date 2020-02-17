# train model
# if you wish to resume from an exists model, uncomment --checkpoint_path and --fastspeech_step
CUDA_VISIBLE_DEVICES=0\
python -u train.py \
--batch_size=32 \
--epochs=10000 \
--lr=0.001 \
--save_step=500 \
--use_gpu=1 \
--use_data_parallel=0 \
--data_path='../../dataset/LJSpeech-1.1' \
--transtts_path='../transformer_tts/checkpoint' \
--transformer_step=160000 \
--save_path='./checkpoint' \
--log_dir='./log' \
--config_path='config/fastspeech.yaml' \
#--checkpoint_path='./checkpoint' \
#--fastspeech_step=97000 \

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi
exit 0