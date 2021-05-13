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

from dataclasses import dataclass
from typing import Optional

from paddle.nn import Layer
from paddle.optimizer import Optimizer
from paddle.io import DataLoader


@dataclass
class UpdaterState:
    iteration: int = 0
    epoch: int = 0


class UpdaterBase(object):
    """An updater is the abstraction of how a model is trained given the
    dataloader and the optimizer.

    The `update_core` method is a step in the training loop with only necessary
    operations (get a batch, forward and backward, update the parameters).

    Other stuffs are made extensions. Visualization, saving, loading and
    periodical validation and evaluation are not considered here.

    But even in such simplist case, things are not that simple. There is an
    attempt to standardize this process and requires only the model and
    dataset and do all the stuffs automatically. But this may hurt flexibility.

    If we assume a batch yield from the dataloader is just the input to the
    model, we will find that some model requires more arguments, or just some
    keyword arguments. But this prevents us from over-simplifying it.

    From another perspective, the batch may includes not just the input, but
    also the target. But the model's forward method may just need the input.
    We can pass a dict or a super-long tuple to the model and let it pick what
    it really needs. But this is an abuse of lazy interface.

    After all, we care about how a model is trained. But just how the model is
    used for inference. We want to control how a model is trained. We just
    don't want to be messed up with other auxiliary code.

    So the best practice is to define a model and define a updater for it.
    """

    def update(self):
        pass

    def update_core(self):
        pass


class StandardUpdater(UpdaterBase):
    """An example of over-simplification. Things may not be that simple, but
    you can subclass it to fit your need.
    """

    def __init__(self,
                 model: Layer,
                 dataloader: DataLoader,
                 optimizer: Optimizer,
                 loss_func=None,
                 auto_new_epoch: bool=True,
                 init_state: Optional[UpdaterState]=None):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.auto_new_epoch = auto_new_epoch
        self.iterator = iter(dataloader)

        if init_state is None:
            self.state = UpdaterState()
        else:
            self.state = init_state

    def update(self):
        self.update_core()
        self.state.iteration += 1

    def new_epoch(self):
        self.iterator = iter(self.dataloader)
        self.state.epoch += 1

    def update_core(self):
        model = self.model
        optimizer = self.optimizer
        loss_func = self.loss_func

        model.train()
        optimizer.clear_grad()

        # fetch a batch
        try:
            batch = next(self.iterator)
        except StopIteration as e:
            if self.auto_new_epoch:
                self.new_epoch()

        # forward
        if self.loss_func is not None:
            loss = loss_func(batch)
        else:
            loss = model(batch)

        # backward
        loss.backward()

        # update parameters
        optimizer.step()
