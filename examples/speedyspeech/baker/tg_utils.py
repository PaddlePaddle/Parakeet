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

import librosa
from praatio import tgio


def validate_textgrid(text_grid, num_samples, sr):
    """Validate Text Grid to make sure that the time interval annotated 
    by the tex grid file does not go beyond the audio file.
    """
    start = text_grid.minTimestamp
    end = text_grid.maxTimestamp

    end_audio = librosa.samples_to_time(num_samples, sr)
    return start == 0.0 and end <= end_audio
