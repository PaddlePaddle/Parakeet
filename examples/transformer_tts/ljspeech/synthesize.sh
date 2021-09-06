#!/bin/bash

python3 synthesize_new.py \
  --transformer-tts-config=conf/default.yaml \
  --transformer-tts-checkpoint=exp_eos_bce_2/default/checkpoints/snapshot_iter_130572.pdz_bak \
  --transformer-tts-stat=dump/train/speech_stats.npy \
  --waveflow-config=waveflow_ljspeech_ckpt_0.3/config.yaml \
  --waveflow-checkpoint=waveflow_ljspeech_ckpt_0.3/step-2000000.pdparams \
  --test-metadata=dump/test/norm/metadata.jsonl \
  --output-dir=exp_eos_bce_2/debug/test \
  --device="gpu" \
  --phones-dict=dump/phone_id_map.txt