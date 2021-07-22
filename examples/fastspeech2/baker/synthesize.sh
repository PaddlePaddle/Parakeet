
#!/bin/bash

python3 synthesize.py \
  --fastspeech2-config=conf/default.yaml \
  --fastspeech2-checkpoint=exp/default/checkpoints/snapshot_iter_62577.pdz \
  --fastspeech2-stat=dump/train/speech_stats.npy \
  --pwg-config=pwg_default.yaml \
  --pwg-params=pwg_generator.pdparams \
  --pwg-stat=pwg_stats.npy \
  --test-metadata=dump/test/norm/metadata.jsonl \
  --output-dir=exp/debug/test \
  --device="gpu" \
  --phones=dump/phone_id_map.txt
