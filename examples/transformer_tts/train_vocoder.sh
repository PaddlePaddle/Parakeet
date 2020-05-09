
# train model
CUDA_VISIBLE_DEVICES=0 \
python -u train_vocoder.py \
--use_gpu=1 \
--data='../../dataset/LJSpeech-1.1' \
--output='./vocoder' \
--config='configs/ljspeech.yaml' \
#--checkpoint='./checkpoint/vocoder/step-100000' \


if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi
exit 0
