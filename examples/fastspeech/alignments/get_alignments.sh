
CUDA_VISIBLE_DEVICES=0 \
python -u get_alignments.py \
--use_gpu=1 \
--output='./alignments' \
--data='../../../dataset/LJSpeech-1.1' \
--config='../../transformer_tts/configs/ljspeech.yaml' \
--checkpoint_transformer='../../transformer_tts/checkpoint/transformer/step-120000' \

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi
exit 0