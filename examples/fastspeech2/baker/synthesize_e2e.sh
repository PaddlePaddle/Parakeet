
#!/bin/bash

python3 synthesize_e2e.py \
  --fastspeech2-config=conf/default.yaml \
  --fastspeech2-checkpoint=exp/default/checkpoints/snapshot_iter_153.pdz \
  --fastspeech2-stat=dump/train/speech_stats.npy \
  --pwg-config=parallel_wavegan_baker_ckpt_0.4/pwg_default.yaml \
  --pwg-params=parallel_wavegan_baker_ckpt_0.4/pwg_generator.pdparams \
  --pwg-stat=parallel_wavegan_baker_ckpt_0.4/pwg_stats.npy \
  --text=../sentences.txt \
  --output-dir=exp/default/test_e2e \
  --device="gpu" \
  --phones-dict=dump/phone_id_map.txt
