
#!/bin/bash

python3 synthesize_e2e.py \
  --fastspeech2-config=conf/default.yaml \
  --fastspeech2-checkpoint=exp/default/checkpoints/snapshot_iter_136017.pdz \
  --fastspeech2-stat=dump/train/speech_stats.npy \
  --pwg-config=pwg_default.yaml \
  --pwg-params=pwg_generator.pdparams \
  --pwg-stat=pwg_stats.npy \
  --text=sentences.txt \
  --output-dir=exp/debug/test_e2e \
  --device="gpu" \
  --phones=dump/phone_id_map.txt
