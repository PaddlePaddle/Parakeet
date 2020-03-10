
# train model
# if you wish to resume from an exists model, uncomment --checkpoint_path and --transformer_step
export CUDA_VISIBLE_DEVICES=2
python -u train_transformer.py \
--batch_size=32 \
--epochs=10000 \
--lr=0.001 \
--save_step=1000 \
--image_step=2000 \
--use_gpu=1 \
--use_data_parallel=0 \
--stop_token=0 \
--data_path='../../dataset/LJSpeech-1.1' \
--save_path='./checkpoint' \
--log_dir='./log' \
--config_path='configs/train_transformer.yaml' \
#--checkpoint_path='./checkpoint' \
#--transformer_step=160000 \

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi
exit 0
