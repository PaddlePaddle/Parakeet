python preprocess.py --rootdir=~/datasets/BZNSYP/ --dumpdir=dump --num_cpu=20
python compute_statistics.py --metadata=dump/train/raw/metadata.jsonl --field-name="feats" --output=dump/train/stats.npy

python normalize.py --metadata=dump/train/raw/metadata.jsonl --dumpdir=dump/train/norm --stats=dump/train/stats.npy
python normalize.py --metadata=dump/dev/raw/metadata.jsonl --dumpdir=dump/dev/norm --stats=dump/train/stats.npy
python normalize.py --metadata=dump/test/raw/metadata.jsonl --dumpdir=dump/test/norm --stats=dump/train/stats.npy
