#!/bin/bash

export MAIN_ROOT=`realpath ${PWD}/../../../`

python3 ${MAIN_ROOT}/utils/pwg_syn.py \
  --config=conf/default.yaml \
  --checkpoint=exp/default/checkpoints/snapshot_iter_400000.pdz\
  --test-metadata=dump/test/norm/metadata.jsonl \
  --output-dir=exp/default/test
