# train model

python -u synthesis.py \
--use_gpu=1 \
--alpha=1.0 \
--checkpoint='./checkpoint/fastspeech/step-120000' \
--config='configs/ljspeech.yaml' \
--config_clarine='../clarinet/configs/config.yaml' \
--checkpoint_clarinet='../clarinet/checkpoint/step-500000' \
--output='./synthesis' \

if [ $? -ne 0 ]; then
    echo "Failed in synthesis!"
    exit 1
fi
exit 0