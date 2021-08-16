#!/bin/bash

# get durations from MFA's result
python3 gen_duration_from_textgrid.py --inputdir ./baker_alignment_tone --output durations.txt

# extract features
python3 preprocess.py --rootdir=~/datasets/BZNSYP/ --dumpdir=dump --dur-file durations.txt --num-cpu 4 --cut-sil True

# # get features' stats(mean and std)
python3 compute_statistics.py --metadata=dump/train/raw/metadata.jsonl --field-name="speech"
python3 compute_statistics.py --metadata=dump/train/raw/metadata.jsonl --field-name="pitch"
python3 compute_statistics.py --metadata=dump/train/raw/metadata.jsonl --field-name="energy"

# normalize and covert phone to id, dev and test should use train's stats
python3 normalize.py --metadata=dump/train/raw/metadata.jsonl --dumpdir=dump/train/norm --speech-stats=dump/train/speech_stats.npy --pitch-stats=dump/train/pitch_stats.npy --energy-stats=dump/train/energy_stats.npy --phones-dict dump/phone_id_map.txt
python3 normalize.py --metadata=dump/dev/raw/metadata.jsonl --dumpdir=dump/dev/norm --speech-stats=dump/train/speech_stats.npy --pitch-stats=dump/train/pitch_stats.npy --energy-stats=dump/train/energy_stats.npy --phones-dict dump/phone_id_map.txt
python3 normalize.py --metadata=dump/test/raw/metadata.jsonl --dumpdir=dump/test/norm --speech-stats=dump/train/speech_stats.npy --pitch-stats=dump/train/pitch_stats.npy --energy-stats=dump/train/energy_stats.npy --phones-dict dump/phone_id_map.txt

