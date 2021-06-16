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

from pathlib import Path
import shutil

import numpy as np
from parakeet.training.checkpoint import KBest, KLatest


def test_kbest():
    def save_fn(path):
        with open(path, 'wt') as f:
            f.write(f"My path is {str(path)}\n")

    K = 1
    kbest_manager = KBest(max_size=K, save_fn=save_fn)
    checkpoint_dir = Path("checkpoints")
    shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True)
    a = np.random.rand(20)
    for i, score in enumerate(a):
        path = checkpoint_dir / f"step_{i}"
        kbest_manager.add_checkpoint(score, path)
    assert len(list(checkpoint_dir.glob("step_*"))) == K


def test_klatest():
    def save_fn(path):
        with open(path, 'wt') as f:
            f.write(f"My path is {str(path)}\n")

    K = 5
    klatest_manager = KLatest(max_size=K, save_fn=save_fn)
    checkpoint_dir = Path("checkpoints")
    shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True)
    for i in range(20):
        path = checkpoint_dir / f"step_{i}"
        klatest_manager.add_checkpoint(path)
    assert len(list(checkpoint_dir.glob("step_*"))) == K
