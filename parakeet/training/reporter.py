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

import contextlib

OBSERVATIONS = None


@contextlib.contextmanager
def scope(observations):
    # make `observation` the target to report to.
    # it is basically a dictionary that stores temporary observations
    global OBSERVATIONS
    old = OBSERVATIONS
    OBSERVATIONS = observations

    try:
        yield
    finally:
        OBSERVATIONS = old


def get_observations():
    global OBSERVATIONS
    return OBSERVATIONS


def report(name, value):
    # a simple function to report named value
    # you can use it everywhere, it will get the default target and writ to it
    # you can think of it as std.out
    observations = get_observations()
    if observations is None:
        return
    else:
        observations[name] = value
