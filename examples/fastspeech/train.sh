# train model
export CUDA_VISIBLE_DEVICES=0
python -u train.py \
--use_gpu=1 \
--data='../../dataset/LJSpeech-1.1' \
--alignments_path='./alignments/alignments.txt' \
--output='./experiment' \
--config='configs/ljspeech.yaml' \
#--checkpoint='./checkpoint/fastspeech/step-120000' \

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi
exit 0