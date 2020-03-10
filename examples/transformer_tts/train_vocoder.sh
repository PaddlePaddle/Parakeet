
# train model
# if you wish to resume from an exists model, uncomment --checkpoint_path and --vocoder_step
CUDA_VISIBLE_DEVICES=0 \
python -u train_vocoder.py \
--batch_size=32 \
--epochs=10000 \
--lr=0.001 \
--save_step=1000 \
--use_gpu=1 \
--use_data_parallel=0 \
--data_path='../../dataset/LJSpeech-1.1' \
--save_path='./checkpoint' \
--log_dir='./log' \
--config_path='configs/train_vocoder.yaml' \
#--checkpoint_path='./checkpoint' \
#--vocoder_step=27000 \


if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi
exit 0
