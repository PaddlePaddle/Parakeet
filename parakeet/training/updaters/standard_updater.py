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

import logging
from dataclasses import dataclass
from typing import Optional
from typing import Dict
from typing import Union

from timer import timer
import paddle
from paddle import Tensor
from paddle.nn import Layer
from paddle.optimizer import Optimizer
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler

from parakeet.training.reporter import report
from parakeet.training.updater import UpdaterBase, UpdaterState


class StandardUpdater(UpdaterBase):
    """An example of over-simplification. Things may not be that simple, but
    you can subclass it to fit your need.
    """

    def __init__(self,
                 model: Layer,
                 optimizer: Optimizer,
                 dataloader: DataLoader,
                 init_state: Optional[UpdaterState]=None):
        # it is designed to hold multiple models
        models = {"main": model}
        self.models: Dict[str, Layer] = models
        self.model = model

        # it is designed to hold multiple optimizers
        optimizers = {"main": optimizer}
        self.optimizer = optimizer
        self.optimizers: Dict[str, Optimizer] = optimizers

        # dataloaders
        self.dataloader = dataloader

        # init state
        if init_state is None:
            self.state = UpdaterState()
        else:
            self.state = init_state

        self.train_iterator = iter(dataloader)

    def update(self):
        self.state.iteration += 1

        # switch to training mode
        for layer in self.models.values():
            layer.train()

        # training for a step is implemented here
        batch = self.read_batch()
        self.update_core(batch)

    def update_core(self, batch):
        """A simple case for a training step. Basic assumptions are:
        Single model;
        Single optimizer;
        A batch from the dataloader is just the input of the model;
        The model return a single loss, or a dict containing serval losses.
        Parameters updates at every batch, no gradient accumulation.
        """
        loss = self.model(*batch)

        if isinstance(loss, Tensor):
            loss_dict = {"main": loss}
        else:
            # Dict[str, Tensor]
            loss_dict = loss
            if "main" not in loss_dict:
                main_loss = 0
                for loss_item in loss.values():
                    main_loss += loss_item
                loss_dict["main"] = main_loss

        for name, loss_item in loss_dict.items():
            report(name, float(loss_item))

        self.optimizer.clear_gradient()
        loss_dict["main"].backward()
        self.optimizer.update()

    def new_epoch(self):
        """Start a new epoch."""
        self.state.epoch += 1

        # NOTE: all batch sampler for distributed training should
        # subclass DistributedBatchSampler and implement `set_epoch` method
        batch_sampler = self.dataloader.batch_sampler
        if isinstance(batch_sampler, DistributedBatchSampler):
            batch_sampler.set_epoch(self.state.epoch)
        self.train_iterator = iter(self.dataloader)

    def read_batch(self):
        """Read a batch from the data loader, auto renew when data is exhausted."""
        with timer() as t:
            try:
                batch = next(self.train_iterator)
            except StopIteration:
                self.new_epoch()
                batch = next(self.train_iterator)
            logging.debug(
                f"Read a batch takes {t.elapse}s.")  # replace it with logging
        return batch

    def state_dict(self):
        """State dict of a Updater, model, optimizer and updater state are included."""
        state_dict = super().state_dict()
        for name, layer in self.models.items():
            state_dict[f"{name}_params"] = layer.state_dict()
        for name, optim in self.optimizers.items():
            state_dict[f"{name}_optimizer"] = optim.state_dict()
        return state_dict

    def set_state_dict(self, state_dict):
        """Set state dict for a Updater. Parameters of models, states for
        optimizers and UpdaterState are restored."""
        for name, layer in self.models.items():
            layer.set_state_dict(state_dict[f"{name}_params"])
        for name, optim in self.optimizers.items():
            optim.set_state_dict(state_dict[f"{name}_optimizer"])
        super().set_state_dict(state_dict)

    def save(self, path):
        """Save Updater state dict."""
        archive = self.state_dict()
        paddle.save(archive, path)

    def load(self, path):
        """Load Updater state dict."""
        archive = paddle.load(path)
        self.set_state_dict(archive)
