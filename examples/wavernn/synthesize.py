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

import argparse
import numpy as np
import soundfile as sf
import os
from pathlib import Path
import paddle
from parakeet.models.wavernn import WaveRNN

from config import get_cfg_defaults


def main(config, args):
    paddle.set_device(args.device)
    model = WaveRNN.from_pretrained(config, args.checkpoint_path)
    model.eval()

    mel_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_path in mel_dir.iterdir():
        mel = np.load(str(file_path))
        mel = paddle.to_tensor(mel).unsqueeze(0)
        # print('mels: {}'.format(mel.shape))
        # print('generate display: {}'.format(args.gen_display))
        audio = model.generate(mels=mel, batched=config.model.gen_batched,
                               target=config.model.target, overlap=config.model.overlap,
                               mu_law=config.data.mu_law, gen_display=args.gen_display)
        audio_path = output_dir / (
            os.path.splitext(file_path.name)[0] + ".wav")
        sf.write(audio_path, audio, config.data.sample_rate)
        print("[synthesize] {} -> {}".format(file_path, audio_path))


if __name__ == "__main__":
    config = get_cfg_defaults()

    parser = argparse.ArgumentParser(
        description="generate mel spectrogram with WaveRNN.")
    parser.add_argument(
        "--config",
        type=str,
        metavar="FILE",
        help="extra config to overwrite the default config")
    parser.add_argument(
        "--checkpoint_path", type=str, help="path of the checkpoint to load.")
    parser.add_argument(
        "--input",
        type=str,
        help="path of directory containing mel spectrogram (in .npy format)")
    parser.add_argument("--output", type=str, help="path to save outputs")
    # parser.add_argument("--gen_display", type=bool, default=True, help="generate display or not")
    parser.add_argument("--gen_display", action="store_true", help="generate display or not")
    parser.add_argument(
        "--device", type=str, default="cpu", help="device type to use.")
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        help="options to overwrite --config file and the default config, passing in KEY VALUE pairs"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print msg")

    args = parser.parse_args()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config)
    print(args)

    main(config, args)
