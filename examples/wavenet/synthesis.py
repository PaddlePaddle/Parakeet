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

from __future__ import division
import os
import ruamel.yaml
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
from paddle import fluid
import paddle.fluid.dygraph as dg

from parakeet.modules.weight_norm import WeightNormWrapper
from parakeet.data import SliceDataset, TransformDataset, DataCargo, SequentialSampler, RandomSampler
from parakeet.models.wavenet import UpsampleNet, WaveNet, ConditionalWavenet
from parakeet.utils.layer_tools import summary

from data import LJSpeechMetaData, Transform, DataCollector
from utils import make_output_tree, valid_model, eval_model, save_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthesize valid data from LJspeech with a wavenet model.")
    parser.add_argument(
        "--data", type=str, help="path of the LJspeech dataset.")
    parser.add_argument("--config", type=str, help="path of the config file.")
    parser.add_argument(
        "--device", type=int, default=-1, help="device to use.")

    parser.add_argument("checkpoint", type=str, help="checkpoint to load.")
    parser.add_argument(
        "output", type=str, default="experiment", help="path to save results.")

    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        config = ruamel.yaml.safe_load(f)

    ljspeech_meta = LJSpeechMetaData(args.data)

    data_config = config["data"]
    sample_rate = data_config["sample_rate"]
    n_fft = data_config["n_fft"]
    win_length = data_config["win_length"]
    hop_length = data_config["hop_length"]
    n_mels = data_config["n_mels"]
    train_clip_seconds = data_config["train_clip_seconds"]
    transform = Transform(sample_rate, n_fft, win_length, hop_length, n_mels)
    ljspeech = TransformDataset(ljspeech_meta, transform)

    valid_size = data_config["valid_size"]
    ljspeech_valid = SliceDataset(ljspeech, 0, valid_size)
    ljspeech_train = SliceDataset(ljspeech, valid_size, len(ljspeech))

    model_config = config["model"]
    n_loop = model_config["n_loop"]
    n_layer = model_config["n_layer"]
    filter_size = model_config["filter_size"]
    context_size = 1 + n_layer * sum([filter_size**i for i in range(n_loop)])
    print("context size is {} samples".format(context_size))
    train_batch_fn = DataCollector(context_size, sample_rate, hop_length,
                                   train_clip_seconds)
    valid_batch_fn = DataCollector(
        context_size, sample_rate, hop_length, train_clip_seconds, valid=True)

    batch_size = data_config["batch_size"]
    train_cargo = DataCargo(
        ljspeech_train,
        train_batch_fn,
        batch_size,
        sampler=RandomSampler(ljspeech_train))

    # only batch=1 for validation is enabled
    valid_cargo = DataCargo(
        ljspeech_valid,
        valid_batch_fn,
        batch_size=1,
        sampler=SequentialSampler(ljspeech_valid))

    make_output_tree(args.output)

    if args.device == -1:
        place = fluid.CPUPlace()
    else:
        place = fluid.CUDAPlace(args.device)

    with dg.guard(place):
        model_config = config["model"]
        upsampling_factors = model_config["upsampling_factors"]
        encoder = UpsampleNet(upsampling_factors)

        n_loop = model_config["n_loop"]
        n_layer = model_config["n_layer"]
        residual_channels = model_config["residual_channels"]
        output_dim = model_config["output_dim"]
        loss_type = model_config["loss_type"]
        log_scale_min = model_config["log_scale_min"]
        decoder = WaveNet(n_loop, n_layer, residual_channels, output_dim,
                          n_mels, filter_size, loss_type, log_scale_min)

        model = ConditionalWavenet(encoder, decoder)
        summary(model)

        model_dict, _ = dg.load_dygraph(args.checkpoint)
        print("Loading from {}.pdparams".format(args.checkpoint))
        model.set_dict(model_dict)

        for layer in model.sublayers():
            if isinstance(layer, WeightNormWrapper):
                layer.remove_weight_norm()

        train_loader = fluid.io.DataLoader.from_generator(
            capacity=10, return_list=True)
        train_loader.set_batch_generator(train_cargo, place)

        valid_loader = fluid.io.DataLoader.from_generator(
            capacity=10, return_list=True)
        valid_loader.set_batch_generator(valid_cargo, place)

        eval_model(model, valid_loader, args.output, sample_rate)
