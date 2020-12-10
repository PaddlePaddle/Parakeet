import argparse

def default_argument_parser():
    parser = argparse.ArgumentParser()

    # data and outpu
    parser.add_argument("--config", metavar="FILE", help="path of the config file to overwrite to default config with.")
    parser.add_argument("--data", metavar="DATA_DIR", help="path to the datatset.")
    parser.add_argument("--output", metavar="OUTPUT_DIR", help="path to save checkpoint and log. If not provided, a directory is created in runs/ to save outputs.")

    # load from saved checkpoint
    parser.add_argument("--checkpoint_path", type=str, help="path of the checkpoint to load")

    # running
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], help="device type to use, cpu and gpu are supported.")
    parser.add_argument("--nprocs", type=int, default=1, help="number of parallel processes to use.")

    # overwrite extra config and default config
    parser.add_argument("--opts", nargs=argparse.REMAINDER, help="options to overwrite --config file and the default config, passing in KEY VALUE pairs")

    return parser
