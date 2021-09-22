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

import numpy as np
import paddle

from parakeet.data.batch import batch_sequences


def collate_baker_examples(examples):
    # fields = ["phones", "tones", "num_phones", "num_frames", "feats", "durations"]
    phones = [np.array(item["phones"], dtype=np.int64) for item in examples]
    tones = [np.array(item["tones"], dtype=np.int64) for item in examples]
    feats = [np.array(item["feats"], dtype=np.float32) for item in examples]
    durations = [
        np.array(item["durations"], dtype=np.int64) for item in examples
    ]
    num_phones = [
        np.array(item["num_phones"], dtype=np.int64) for item in examples
    ]
    num_frames = [
        np.array(item["num_frames"], dtype=np.int64) for item in examples
    ]

    phones = batch_sequences(phones)
    tones = batch_sequences(tones)
    feats = batch_sequences(feats)
    durations = batch_sequences(durations)

    # convert each batch to paddle.Tensor
    phones = paddle.to_tensor(phones)
    tones = paddle.to_tensor(tones)
    feats = paddle.to_tensor(feats)
    durations = paddle.to_tensor(durations)
    num_phones = paddle.to_tensor(num_phones)
    num_frames = paddle.to_tensor(num_frames)
    batch = {
        "phones": phones,
        "tones": tones,
        "num_phones": num_phones,
        "num_frames": num_frames,
        "feats": feats,
        "durations": durations,
    }
    return batch
