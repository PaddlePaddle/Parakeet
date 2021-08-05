#!/bin/bash

# test g2p
echo "Start test g2p."
python3 test_g2p.py --root-dir=~/datasets/BZNSYP
# test text normalization
echo "Start test text normalization."
python3 test_textnorm.py --test-file=data/textnorm_test_cases.txt