# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Normalize feature files and dump them."""

import argparse
import logging
import os
from copy import copy
from operator import itemgetter
from pathlib import Path

import numpy as np
import yaml
import jsonlines
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from parakeet.frontend.vocab import Vocab
from parakeet.datasets.data_table import DataTable

from config import get_cfg_default


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Normalize dumped raw features (See detail in parallel_wavegan/bin/normalize.py)."
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="directory including feature files to be normalized. "
        "you need to specify either *-scp or rootdir.")
    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump normalized feature files.")
    parser.add_argument(
        "--stats", type=str, required=True, help="statistics file.")
    parser.add_argument(
        "--phones",
        type=str,
        default="phones.txt",
        help="phone vocabulary file.")
    parser.add_argument(
        "--tones", type=str, default="tones.txt", help="tone vocabulary file.")
    parser.add_argument(
        "--config", type=str, help="yaml format configuration file.")
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)")
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
        )
        logging.warning('Skip DEBUG/INFO messages')

    # load config
    config = get_cfg_default()
    if args.config:
        config.merge_from_file(args.config)

    # check directory existence
    dumpdir = Path(args.dumpdir).resolve()
    dumpdir.mkdir(parents=True, exist_ok=True)

    # get dataset
    with jsonlines.open(args.metadata, 'r') as reader:
        metadata = list(reader)
    metadata_dir = Path(args.metadata).parent
    for item in metadata:
        item["feats"] = str(metadata_dir / item["feats"])

    dataset = DataTable(metadata, converters={'feats': np.load, })
    logging.info(f"The number of files = {len(dataset)}.")

    # restore scaler
    scaler = StandardScaler()
    scaler.mean_ = np.load(args.stats)[0]
    scaler.scale_ = np.load(args.stats)[1]

    # from version 0.23.0, this information is needed
    scaler.n_features_in_ = scaler.mean_.shape[0]

    with open(args.phones, 'rt') as f:
        phones = [line.strip() for line in f.readlines()]

    with open(args.tones, 'rt') as f:
        tones = [line.strip() for line in f.readlines()]
    voc_phones = Vocab(phones, start_symbol=None, end_symbol=None)
    voc_tones = Vocab(tones, start_symbol=None, end_symbol=None)

    # process each file
    output_metadata = []

    for item in tqdm(dataset):
        utt_id = item['utt_id']
        mel = item['feats']
        # normalize
        mel = scaler.transform(mel)

        # save
        mel_path = dumpdir / f"{utt_id}-feats.npy"
        np.save(mel_path, mel.astype(np.float32), allow_pickle=False)
        phone_ids = [voc_phones.lookup(p) for p in item['phones']]
        tone_ids = [voc_tones.lookup(t) for t in item['tones']]
        output_metadata.append({
            'utt_id': utt_id,
            'phones': phone_ids,
            'tones': tone_ids,
            'num_phones': item['num_phones'],
            'num_frames': item['num_frames'],
            'durations': item['durations'],
            'feats': str(mel_path.relative_to(dumpdir)),
        })
    output_metadata.sort(key=itemgetter('utt_id'))
    output_metadata_path = Path(args.dumpdir) / "metadata.jsonl"
    with jsonlines.open(output_metadata_path, 'w') as writer:
        for item in output_metadata:
            writer.write(item)
    logging.info(f"metadata dumped into {output_metadata_path}")


if __name__ == "__main__":
    main()
