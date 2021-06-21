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

from typing import Union
from pathlib import Path

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

    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.checkpoint_dir = Path(checkpoint_dir)

    @rank_zero_only
    def __call__(self, trainer: Trainer):
        iteration = trainer.updater.state.iteration
        path = self.checkpoint_dir / f"step_{iteration}.pdz"
        trainer.updater.save(str(path))
