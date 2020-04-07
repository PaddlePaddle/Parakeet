# train model

python -u synthesis.py \
--use_gpu=1 \
--alpha=1.0 \
--checkpoint_path='checkpoint/' \
--fastspeech_step=89000 \
--log_dir='./log' \
--config_path='configs/synthesis.yaml' \

if [ $? -ne 0 ]; then
    echo "Failed in synthesis!"
    exit 1
fi
exit 0