import os
import random
from pprint import pprint

import jsonargparse
import numpy as np
import paddle.fluid.dygraph as dg
from paddle import fluid

import utils
from waveflow import WaveFlow


def add_options_to_parser(parser):
    parser.add_argument(
        '--model',
        type=str,
        default='waveflow',
        help="general name of the model")
    parser.add_argument(
        '--name', type=str, help="specific name of the training model")
    parser.add_argument(
        '--root', type=str, help="root path of the LJSpeech dataset")

    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=True,
        help="option to use gpu training")

    parser.add_argument(
        '--iteration',
        type=int,
        default=None,
        help=("which iteration of checkpoint to load, "
              "default to load the latest checkpoint"))
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help="path of the checkpoint to load")


def benchmark(config):
    pprint(jsonargparse.namespace_to_dict(config))

    # Get checkpoint directory path.
    run_dir = os.path.join("runs", config.model, config.name)
    checkpoint_dir = os.path.join(run_dir, "checkpoint")

    # Configurate device.
    place = fluid.CUDAPlace(0) if config.use_gpu else fluid.CPUPlace()

    with dg.guard(place):
        # Fix random seed.
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        fluid.default_startup_program().random_seed = seed
        fluid.default_main_program().random_seed = seed
        print("Random Seed: ", seed)

        # Build model.
        model = WaveFlow(config, checkpoint_dir)
        model.build(training=False)

        # Run model inference.
        model.benchmark()


if __name__ == "__main__":
    # Create parser.
    parser = jsonargparse.ArgumentParser(
        description="Synthesize audio using WaveNet model",
        formatter_class='default_argparse')
    add_options_to_parser(parser)
    utils.add_config_options_to_parser(parser)

    # Parse argument from both command line and yaml config file.
    # For conflicting updates to the same field,
    # the preceding update will be overwritten by the following one.
    config = parser.parse_args()
    benchmark(config)
