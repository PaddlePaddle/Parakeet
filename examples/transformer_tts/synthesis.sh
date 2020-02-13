
# train model
#CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -u synthesis.py \
--max_len=50 \
--transformer_step=160000 \
--postnet_step=70000 \
--use_gpu=1
--checkpoint_path='./checkpoint' \
--log_dir='./log' \
--sample_path='./sample' \
--config_path='config/synthesis.yaml' \

if [ $? -ne 0 ]; then
    echo "Failed in training!"
    exit 1
fi
exit 0