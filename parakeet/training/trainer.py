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
import tqdm
from dataclasses import dataclass

from parakeet.training.trigger import get_trigger, IntervalTrigger
from parakeet.training.updater import UpdaterBase
from parakeet.training.reporter import scope


class ExtensionEntry(object):
    def __init__(self, extension, trigger, priority):
        self.extension = extension
        self.trigger = trigger
        self.priority = priority


class Trainer(object):
    def __init__(self,
                 updater: UpdaterBase,
                 stop_trigger=None,
                 out='result',
                 extensions=None):
        self.updater = updater
        self.extensions = {}
        self.stop_trigger = get_trigger(stop_trigger)
        self.out = Path(out)
        self.observation = {}

    def setup(self):
        pass

    def extend(self, extension, name=None, trigger=None, priority=None):
        trigger = get_trigger(trigger)

        ordinal = 0
        modified_name = name
        while name in self.extensions:
            ordinal += 1
            modified_name = f"{name}_{ordinal}"

        self.extensions[modified_name] = ExtensionEntry(extension, trigger,
                                                        priority)

    def run(self):
        # sort extensions by priorities once
        extension_order = sorted(
            self.extensions.keys(),
            key=lambda name: self.extensions[name].priority,
            reverse=True)
        extensions = [(name, self.extensions[name])
                      for name in extension_order]

        update = self.updater.update
        stop_trigger = self.stop_trigger

        # TODO(chenfeiyu): display progress bar correctly
        # if the trainer is controlled by epoch: use 2 progressbars
        # if the trainer is controlled by iteration: use 1 progressbar
        if isinstance(stop_trigger, IntervalTrigger):
            if stop_trigger.unit is 'epoch':
                max_epoch = self.stop_trigger.period
            else:
                max_iteration = self.stop_trigger.period

        while not stop_trigger(self):
            self.observation = {}
            # set observation as the report target
            # you can use report freely in Updater.update()

            # updating parameters and state
            with scope(self.observation):
                update()

            # execute extension when necessary
            for name, entry in extensions:
                if entry.trigger(self):
                    entry.extension(self)
