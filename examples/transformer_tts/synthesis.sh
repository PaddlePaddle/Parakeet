
# train model
CUDA_VISIBLE_DEVICES=0 \
python -u synthesis.py \
--max_len=300 \
--use_gpu=1 \
--output='./synthesis' \
--config='configs/ljspeech.yaml' \
--checkpoint_transformer='./checkpoint/transformer/step-120000' \
--checkpoint_vocoder='./checkpoint/vocoder/step-100000' \

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi
exit 0
