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
import numpy as np
import paddle

from parakeet.data.batch import batch_sequences


def collate_ljspeech_examples(examples):
    # fields = ["text", "text_lengths", "speech", "speech_lengths"]
    text = [np.array(item["text"], dtype=np.int64) for item in examples]
    speech = [np.array(item["speech"], dtype=np.float32) for item in examples]
    text_lengths = np.array([item["text_lengths"] for item in examples])
    speech_lengths = np.array([item["speech_lengths"] for item in examples])

    text = batch_sequences(text)
    speech = batch_sequences(speech)

    # convert each batch to paddle.Tensor
    text = paddle.to_tensor(text)
    speech = paddle.to_tensor(speech)
    text_lengths = paddle.to_tensor(text_lengths)
    speech_lengths = paddle.to_tensor(speech_lengths)

    batch = {
        "text": text,
        "text_lengths": text_lengths,
        "speech": speech,
        "speech_lengths": speech_lengths,
    }
    return batch
