FLAGS_cudnn_exhaustive_search=true \
FLAGS_conv_workspace_size_limit=4000 \
CUDA_VISIBLE_DEVICES="3" \
python train.py \
    --train-metadata=dump/train/norm/metadata.jsonl \
    --dev-metadata=dump/dev/norm/metadata.jsonl \
    --config=conf/debug.yaml \
    --output-dir=exp/debug \
    --nprocs=1