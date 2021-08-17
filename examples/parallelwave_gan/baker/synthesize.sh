python3 synthesize.py \
  --config=conf/default.yaml \
  --checkpoint=exp/default/checkpoints/snapshot_iter_220000.pdz \
  --test-metadata=dump/test/norm/metadata.jsonl \
  --output-dir=exp/debug/test
