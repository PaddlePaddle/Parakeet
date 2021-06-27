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
from collections import OrderedDict
from typing import Callable, Union, List

import tqdm

from parakeet.training.trigger import get_trigger, IntervalTrigger
from parakeet.training.updater import UpdaterBase
from parakeet.training.reporter import scope
from parakeet.training.extension import Extension, PRIORITY_READER


class _ExtensionEntry(object):
    def __init__(self, extension, trigger, priority):
        self.extension = extension
        self.trigger = trigger
        self.priority = priority


class Trainer(object):
    def __init__(self,
                 updater: UpdaterBase,
                 stop_trigger: Callable=None,
                 out: Union[str, Path]='result',
                 extensions: List[Extension]=None):
        self.updater = updater
        self.extensions = OrderedDict()
        self.stop_trigger = get_trigger(stop_trigger)
        self.out = Path(out)
        self.observation =...

        self._done = False
        if extensions:
            for ext in extensions:
                self.extend(ext)

    @property
    def is_before_training(self):
        return self.updater.state.iteration == 0

    def extend(self, extension, name=None, trigger=None, priority=None):
        # get name for the extension
        # argument \
        # -> extention's name \
        # -> default_name (class name, when it is an object) \
        # -> function name when it is a function \
        # -> error

        if name is None:
            name = getattr(extension, 'name', None)
            if name is None:
                name = getattr(extenion, 'default_name', None)
                if name is None:
                    name = getattr(extension, '__name__', None)
                    if name is None:
                        raise ValueError(
                            "Name is not given for the extension.")
        if name == 'training':
            raise ValueError("training is a reserved name.")

        if trigger is None:
            trigger = getattr(extension, 'trigger', (1, 'iteration'))
        trigger = get_trigger(trigger)

        if priority is None:
            priority = getattr(extension, 'priority', PRIORITY_READER)

        # add suffix to avoid nameing conflict
        ordinal = 0
        modified_name = name
        while modified_name in self.extensions:
            ordinal += 1
            modified_name = f"{name}_{ordinal}"
        extension.name = modified_name

        self.extensions[modified_name] = _ExtensionEntry(extension, trigger,
                                                         priority)

    def get_extension(self, name):
        """get extension by name."""
        extensions = self.extensions
        if name in extensions:
            return extensions[name].extension
        else:
            raise ValueError(f'extension {name} not found')

    def run(self):
        if self._done:
            raise RuntimeError("Training is already done!.")

        self.out.mkdir(parents=True, exist_ok=True)

        # sort extensions by priorities once
        extension_order = sorted(
            self.extensions.keys(),
            key=lambda name: self.extensions[name].priority,
            reverse=True)
        extensions = [(name, self.extensions[name])
                      for name in extension_order]

        print("initializing")
        for name, entry in extensions:
            if hasattr(entry.extension, "initialize"):
                entry.extension.initialize(self)

        update = self.updater.update  # training step
        stop_trigger = self.stop_trigger

        # TODO(chenfeiyu): display progress bar correctly
        # if the trainer is controlled by epoch: use 2 progressbars
        # if the trainer is controlled by iteration: use 1 progressbar
        if isinstance(stop_trigger, IntervalTrigger):
            if stop_trigger.unit is 'epoch':
                max_epoch = self.stop_trigger.period
            else:
                max_iteration = self.stop_trigger.period

        p = tqdm.tqdm()

        while True:
            self.observation = {}
            # set observation as the report target
            # you can use report freely in Updater.update()

            # updating parameters and state
            with scope(self.observation):
                update()
                p.update()
                print(self.observation)

                # execute extension when necessary
                for name, entry in extensions:
                    if entry.trigger(self):
                        entry.extension(self)

            if stop_trigger(self):
                print("Training Done!")
                break
