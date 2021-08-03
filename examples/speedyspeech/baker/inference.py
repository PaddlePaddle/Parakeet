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

import argparse
from pathlib import Path

import numpy as np
from paddle import inference
import soundfile as sf

from frontend import text_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Paddle Infernce with speedyspeech & parallel wavegan.")
    parser.add_argument(
        "--inference-dir", type=str, help="dir to save inference models")
    parser.add_argument(
        "--text",
        type=str,
        help="text to synthesize, a 'utt_id sentence' pair per line")
    parser.add_argument("--output-dir", type=str, help="output dir")

    args = parser.parse_args()

    speedyspeech_config = inference.Config(
        str(Path(args.inference_dir) / "speedyspeech.pdmodel"),
        str(Path(args.inference_dir) / "speedyspeech.pdiparams"))
    speedyspeech_config.enable_use_gpu(100, 0)
    speedyspeech_config.enable_memory_optim()
    speedyspeech_predictor = inference.create_predictor(speedyspeech_config)

    pwg_config = inference.Config(
        str(Path(args.inference_dir) / "pwg.pdmodel"),
        str(Path(args.inference_dir) / "pwg.pdiparams"))
    pwg_config.enable_use_gpu(100, 0)
    pwg_config.enable_memory_optim()
    pwg_predictor = inference.create_predictor(pwg_config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sentences = []
    with open(args.text, 'rt') as f:
        for line in f:
            utt_id, sentence = line.strip().split()
            sentences.append((utt_id, sentence))

    for utt_id, sentence in sentences:
        phones, tones = text_analysis(sentence)
        phones = phones.numpy()
        tones = tones.numpy()

        input_names = speedyspeech_predictor.get_input_names()
        phones_handle = speedyspeech_predictor.get_input_handle(input_names[0])
        tones_handle = speedyspeech_predictor.get_input_handle(input_names[1])

        phones_handle.reshape(phones.shape)
        phones_handle.copy_from_cpu(phones)
        tones_handle.reshape(tones.shape)
        tones_handle.copy_from_cpu(tones)

        speedyspeech_predictor.run()
        output_names = speedyspeech_predictor.get_output_names()
        output_handle = speedyspeech_predictor.get_output_handle(output_names[
            0])
        output_data = output_handle.copy_to_cpu()

        input_names = pwg_predictor.get_input_names()
        mel_handle = pwg_predictor.get_input_handle(input_names[0])
        mel_handle.reshape(output_data.shape)
        mel_handle.copy_from_cpu(output_data)

        pwg_predictor.run()
        output_names = pwg_predictor.get_output_names()
        output_handle = pwg_predictor.get_output_handle(output_names[0])
        wav = output_data = output_handle.copy_to_cpu()

        sf.write(output_dir / (utt_id + ".wav"), wav, samplerate=24000)
        print(f"{utt_id} done!")


if __name__ == "__main__":
    main()
