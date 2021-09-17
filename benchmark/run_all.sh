#!/usr/bin/env bash

# 提供可稳定复现性能的脚本，默认在标准docker环境内py37执行： paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  paddle=2.1.2  py=37
# 执行目录：需说明
cd ../
# 1 安装该模型需要的依赖 (如需开启优化策略请注明)
# sudo apt-get install libsndfile1
pip install -e .
# 2 拷贝该模型需要数据、预训练模型
# 下载 baker 数据集到 home 目录下并解压缩到 home 目录下
# wget https://paddlespeech.bj.bcebos.com/datasets/BZNSYP.rar

# mkdir BZNSYP
# unrar x BZNSYP.rar BZNSYP
# # 数据预处理
# python examples/parallelwave_gan/baker/preprocess.py --rootdir=~/datasets/BZNSYP/ --dumpdir=dump --num_cpu=20
# python examples/parallelwave_gan/baker/compute_statistics.py --metadata=dump/train/raw/metadata.jsonl --field-name="feats" --dumpdir=dump/train

# python examples/parallelwave_gan/baker/normalize.py --metadata=dump/train/raw/metadata.jsonl --dumpdir=dump/train/norm --stats=dump/train/stats.npy
# python examples/parallelwave_gan/baker/normalize.py --metadata=dump/dev/raw/metadata.jsonl --dumpdir=dump/dev/norm --stats=dump/train/stats.npy
# python examples/parallelwave_gan/baker/normalize.py --metadata=dump/test/raw/metadata.jsonl --dumpdir=dump/test/norm --stats=dump/train/stats.npy

# 3 批量运行（如不方便批量，1，2需放到单个模型中）
 
model_mode_list=(pwg)
fp_item_list=(fp32)
bs_item_list=(6 26)
for model_mode in ${model_mode_list[@]}; do
      for fp_item in ${fp_item_list[@]}; do
          for bs_item in ${bs_item_list[@]}; do
            echo "index is speed, 1gpus, begin, ${model_name}"
            run_mode=sp
            CUDA_VISIBLE_DEVICES=5 FLAGS_cudnn_exhaustive_search=true FLAGS_conv_workspace_size_limit=4000 bash benchmark/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 500 ${model_mode}     #  (5min)
            sleep 60
            # echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
            # run_mode=mp
            # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 FLAGS_cudnn_exhaustive_search=true FLAGS_conv_workspace_size_limit=4000 bash benchmark/run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} 500 ${model_mode} 
            # sleep 60
            done
      done
done