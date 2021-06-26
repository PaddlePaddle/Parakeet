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

from typing import Union, List, Dict, Any
from pathlib import Path
import jsonlines
import os
from datetime import datetime
import logging

from parakeet.utils.mp_tools import rank_zero_only
from parakeet.training.trainer import Trainer


class Snapshot(object):
    """An extension to make snapshot of the updater object inside
    the trainer. It is done by calling the updater's `save` method.

    An Updater save its state_dict by default, which contains the
    updater state, (i.e. epoch and iteration) and all the model
    parameters and optimizer states. If the updater inside the trainer
    subclasses StandardUpdater, everything is good to go.

    Parameters
    ----------
    checkpoint_dir : Union[str, Path]
        The directory to save checkpoints into.
    """

    def __init__(self, max_size: int=5):
        self.records: List[Dict[str, Any]] = []
        self.max_size = max_size
        self._save_all = (max_size == -1)
        self.save_fn =...
        self.del_fn =...
        self.checkpoint_dir =...

    def initialize(self, trainer):
        """setting up this extention."""
        self.save_fn = trainer.updater.save
        self.del_fn = os.remove
        self.checkpoint_dir = trainer.out / "checkpoints"

    def full(self):
        return (not self._save_all) and len(self.records) >= self.max_size

    @rank_zero_only
    def save_checkpoint_and_update(self, trainer):
        iteration = trainer.updater.state.iteration
        path = self.checkpoint_dir / f"snapshot_iter_{iteration}.pdz"

        # remove the earist
        if self.full():
            eariest_record = self.records[0]
            self.del_fn(eariest_record["path"])
            self.records.pop(0)

        # add the new one
        self.save_fn(path)
        record = {
            "time": str(datetime.now()),
            'path': str(path),
            'iteration': iteration
        }
        self.records.append(record)

        # update the record
        with jsonlines.open(self.checkpoint_dir / "records.jsonl", 'w') as f:
            for record in self.records:
                f.write(record)

    def __call__(self, trainer):
        self.save_checkpoint_and_update(trainer)
