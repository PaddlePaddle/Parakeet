# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


def get_alignment(attn_probs, mel_lens, n_head):
    max_F = 0
    assert attn_probs[0].shape[0] % n_head == 0
    batch_size = int(attn_probs[0].shape[0] // n_head)
    for i in range(len(attn_probs)):
        multi_attn = attn_probs[i].numpy()
        for j in range(n_head):
            attn = multi_attn[j * batch_size:(j + 1) * batch_size]
            F = score_F(attn)
            if max_F < F:
                max_F = F
                max_attn = attn
    alignment = compute_duration(max_attn, mel_lens)
    return alignment, max_attn


def score_F(attn):
    max = np.max(attn, axis=-1)
    mean = np.mean(max)
    return mean


def compute_duration(attn, mel_lens):
    alignment = np.zeros([attn.shape[2]])
    #for i in range(attn.shape[0]):
    for j in range(mel_lens):
        max_index = np.argmax(attn[0, j])
        alignment[max_index] += 1

    return alignment
