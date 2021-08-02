
#!/bin/bash

python3 synthesize.py \
  --fastspeech2-config=conf/default.yaml \
  --fastspeech2-checkpoint=exp/default/checkpoints/snapshot_iter_62577.pdz \
  --fastspeech2-stat=dump/train/speech_stats.npy \
  --pwg-config=parallel_wavegan_baker_ckpt_1.0/pwg_default.yaml \
  --pwg-params=parallel_wavegan_baker_ckpt_1.0/pwg_generator.pdparams \
  --pwg-stat=parallel_wavegan_baker_ckpt_1.0/pwg_stats.npy \
  --test-metadata=dump/test/norm/metadata.jsonl \
  --output-dir=exp/debug/test \
  --device="gpu" \
  --phones=dump/phone_id_map.txt
