python synthesize_e2e.py \
  --speedyspeech-config=conf/default.yaml \
  --speedyspeech-checkpoint=exp/debug/checkpoints/snapshot_iter_91800.pdz \
  --speedyspeech-stat=dump/train/stats.npy \
  --pwg-config=../../parallelwave_gan/baker/conf/default.yaml \
  --pwg-params=../../parallelwave_gan/baker/converted.pdparams \
  --pwg-stat=../../parallelwave_gan/baker/dump/train/stats.npy \
  --text=sentences.txt \
  --output-dir=exp/e2e \
  --device="gpu"
