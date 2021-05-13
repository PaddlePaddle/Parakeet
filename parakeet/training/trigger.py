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


class IntervalTrigger(object):
    def __init__(self, period: int, unit: str):
        if unit not in ("iteration", "epoch"):
            raise ValueError("unit should be 'iteration' or 'epoch'")
        self.period = period
        self.unit = unit

    def __call__(self, trainer):
        state = trainer.updater.state
        if self.unit == "epoch":
            fire = not (state.epoch % self.period)
        else:
            fire = not (state.iteration % self.iteration)
        return fire


def never_file_trigger(trainer):
    return False


def get_trigger(trigger):
    if trigger is None:
        return never_file_trigger
    if callable(trigger):
        return trigger
    else:
        trigger = IntervalTrigger(*trigger)
        return trigger
