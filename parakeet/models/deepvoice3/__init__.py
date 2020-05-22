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

from parakeet.models.deepvoice3.encoder import Encoder, ConvSpec
from parakeet.models.deepvoice3.decoder import Decoder, WindowRange
from parakeet.models.deepvoice3.converter import Converter
from parakeet.models.deepvoice3.loss import TTSLoss
from parakeet.models.deepvoice3.model import DeepVoice3
