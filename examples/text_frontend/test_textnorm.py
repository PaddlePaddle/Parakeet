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

import argparse
import re
from pathlib import Path

from parakeet.frontend.cn_normalization.text_normlization import TextNormalizer
from parakeet.utils.error_rate import cer


# delete english characters
# e.g. "你好aBC" -> "你 好"
def del_en_add_space(input: str):
    output = re.sub('[a-zA-Z]', '', input)
    output = [char + " " for char in output]
    output = "".join(output).strip()
    return output


def get_avg_cer(test_file, text_normalizer):
    cer_list = []
    for line in open(test_file, "r"):
        line = line.strip()
        raw_text, gt_text = line.split("|")
        textnorm_text = text_normalizer.normalize_sentence(raw_text)
        gt_text = del_en_add_space(gt_text)
        textnorm_text = del_en_add_space(textnorm_text)
        single_cer = cer(gt_text, textnorm_text)
        cer_list.append(single_cer)
    return sum(cer_list) / len(cer_list)


def main():
    parser = argparse.ArgumentParser(description="text normalization example.")
    parser.add_argument(
        "--test-file",
        default=None,
        type=str,
        help="path of text normalization test file.")

    args = parser.parse_args()
    test_file = Path(args.test_file).expanduser()
    text_normalizer = TextNormalizer()
    avg_cer = get_avg_cer(test_file, text_normalizer)
    print("The avg CER of text normalization is:", avg_cer)


if __name__ == "__main__":
    main()
