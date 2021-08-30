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


def collate_aishell3_examples(examples):
    # fields = ["text", "text_lengths", "speech", "speech_lengths", "durations", "pitch", "energy", "spk_id"]
    text = [np.array(item["text"], dtype=np.int64) for item in examples]
    speech = [np.array(item["speech"], dtype=np.float32) for item in examples]
    pitch = [np.array(item["pitch"], dtype=np.float32) for item in examples]
    energy = [np.array(item["energy"], dtype=np.float32) for item in examples]
    durations = [
        np.array(item["durations"], dtype=np.int64) for item in examples
    ]
    text_lengths = np.array([item["text_lengths"] for item in examples])
    speech_lengths = np.array([item["speech_lengths"] for item in examples])
    spk_id = np.array([item["spk_id"] for item in examples])

    text = batch_sequences(text)
    pitch = batch_sequences(pitch)
    speech = batch_sequences(speech)
    durations = batch_sequences(durations)
    energy = batch_sequences(energy)

    # convert each batch to paddle.Tensor
    text = paddle.to_tensor(text)
    pitch = paddle.to_tensor(pitch)
    speech = paddle.to_tensor(speech)
    durations = paddle.to_tensor(durations)
    energy = paddle.to_tensor(energy)
    text_lengths = paddle.to_tensor(text_lengths)
    speech_lengths = paddle.to_tensor(speech_lengths)
    spk_id = paddle.to_tensor(spk_id)

    batch = {
        "text": text,
        "text_lengths": text_lengths,
        "durations": durations,
        "speech": speech,
        "speech_lengths": speech_lengths,
        "pitch": pitch,
        "energy": energy,
        "spk_id": spk_id
    }
    return batch
