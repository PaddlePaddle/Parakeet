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


def add_config_options_to_parser(parser):
    parser.add_argument(
        '--config_path',
        type=str,
        default='configs/fastspeech.yaml',
        help="the yaml config file path.")
    parser.add_argument(
        '--batch_size', type=int, default=32, help="batch size for training.")
    parser.add_argument(
        '--epochs',
        type=int,
        default=10000,
        help="the number of epoch for training.")
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help="the learning rate for training.")
    parser.add_argument(
        '--save_step',
        type=int,
        default=500,
        help="checkpointing interval during training.")
    parser.add_argument(
        '--fastspeech_step',
        type=int,
        default=70000,
        help="Global step to restore checkpoint of fastspeech.")
    parser.add_argument(
        '--use_gpu',
        type=int,
        default=1,
        help="use gpu or not during training.")
    parser.add_argument(
        '--use_data_parallel',
        type=int,
        default=0,
        help="use data parallel or not during training.")
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help="The hyperparameter to determine the length of the expanded sequence \
                mel, thereby controlling the voice speed.")

    parser.add_argument(
        '--data_path',
        type=str,
        default='./dataset/LJSpeech-1.1',
        help="the path of dataset.")
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help="the path to load checkpoint or pretrain model.")
    parser.add_argument(
        '--save_path',
        type=str,
        default='./checkpoint',
        help="the path to save checkpoint.")
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./log',
        help="the directory to save tensorboard log.")
    parser.add_argument(
        '--sample_path',
        type=str,
        default='./sample',
        help="the directory to save audio sample in synthesis.")
    parser.add_argument(
        '--transtts_path',
        type=str,
        default='../transformer_tts/checkpoint',
        help="the directory to load pretrain transformerTTS model.")
    parser.add_argument(
        '--transformer_step',
        type=int,
        default=160000,
        help="the step to load transformerTTS model.")
