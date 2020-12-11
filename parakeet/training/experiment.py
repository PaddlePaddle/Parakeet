import time
import logging
from pathlib import Path
import numpy as np
import paddle
from paddle import distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler
from tensorboardX import SummaryWriter
from collections import defaultdict

import parakeet
from parakeet.utils import checkpoint, mp_tools

class ExperimentBase(object):
    """
    An experiment template in order to structure the training code and take care of saving, loading, logging, visualization stuffs. It's intended to be flexible and simple. 
    
    So it only handles output directory (create directory for the outut, create a checkpoint directory, dump the config in use and create visualizer and logger)in a standard way without restricting the input/output protocols of the model and dataloader. It leaves the main part for the user to implement their own(setup the model, criterion, optimizer, defaine a training step, define a validation function and customize all the text and visual logs).

    It does not save too much boilerplate code. The users still have to write the forward/backward/update mannually, but they are free to add non-standard behaviors if needed.

    We have some conventions to follow.
    1. Experiment should have `.model`, `.optimizer`, `.train_loader` and `.valid_loader`, `.config`, `.args` attributes.
    2. The config should have a `.training` field, which has `valid_interval`, `save_interval` and `max_iteration` keys. It is used as the trigger to invoke validation, checkpointing and stop of the experiment.
    3. There are three method, namely `train_batch`, `valid`, `setup_model` and `setup_dataloader` that should be implemented.

    Feel free to add/overwrite other methods and standalone functions if you need.

    Examples:
    --------
    def main_sp(config, args):
        exp = Experiment(config, args)
        exp.setup()
        exp.run()

    def main(config, args):
        if args.nprocs > 1 and args.device == "gpu":
            dist.spawn(main_sp, args=(config, args), nprocs=args.nprocs)
        else:
            main_sp(config, args)

    if __name__ == "__main__":
        config = get_cfg_defaults()
        parser = default_argument_parser()
        args = parser.parse_args()
        if args.config: 
            config.merge_from_file(args.config)
        if args.opts:
            config.merge_from_list(args.opts)
        config.freeze()
        print(config)
        print(args)

        main(config, args)

    """
    def __init__(self, config, args):
        self.config = config
        self.args = args

    def setup(self):
        if self.parallel:
            self.init_parallel()

        self.setup_output_dir()
        self.dump_config()
        self.setup_visualizer()
        self.setup_logger()
        self.setup_checkpointer()
        
        self.setup_dataloader()
        self.setup_model()

        self.iteration = 0
        self.epoch = 0

    @property
    def parallel(self):
        return self.args.device == "gpu" and self.args.nprocs > 1

    def init_parallel(self):
        dist.init_parallel_env()

    def save(self):
        checkpoint.save_parameters(
            self.checkpoint_dir, self.iteration, self.model, self.optimizer)

    def resume_or_load(self):
        iteration = checkpoint.load_parameters(
            self.model, 
            self.optimizer, 
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_path=self.args.checkpoint_path)
        self.iteration = iteration

    def read_batch(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.new_epoch()
            batch = next(self.iterator)
        return batch

    def new_epoch(self):
        self.epoch += 1
        if self.parallel:
            self.train_loader.batch_sampler.set_epoch(self.epoch)
        self.iterator = iter(self.train_loader)

    def train(self):
        self.new_epoch()
        while self.iteration < self.config.training.max_iteration:
            self.iteration += 1
            self.train_batch()

            if self.iteration % self.config.training.valid_interval == 0:
                self.valid()
        
            if self.iteration % self.config.training.save_interval == 0:
                self.save()
    
    def run(self):
        self.resume_or_load()
        try:
            self.train()
        except KeyboardInterrupt:
            self.save()
            exit(-1)
    
    @mp_tools.rank_zero_only
    def setup_output_dir(self):
        # output dir
        output_dir = Path(self.args.output).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = output_dir
    
    @mp_tools.rank_zero_only
    def setup_checkpointer(self):
        # checkpoint dir
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        self.checkpoint_dir = checkpoint_dir

    @mp_tools.rank_zero_only
    def setup_visualizer(self):
        # visualizer
        visualizer = SummaryWriter(logdir=str(self.output_dir))

        self.visualizer = visualizer

    def setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel("INFO")
        logger.addHandler(logging.StreamHandler())
        log_file = self.output_dir / 'worker_{}.log'.format(dist.get_rank())
        logger.addHandler(logging.FileHandler(str(log_file)))

        self.logger = logger

    @mp_tools.rank_zero_only
    def dump_config(self):
        with open(self.output_dir / "config.yaml", 'wt') as f: 
            print(self.config, file=f)

    def train_batch(self):
        raise NotImplementedError("train_batch should be implemented.")

    @mp_tools.rank_zero_only
    @paddle.no_grad()
    def valid(self):
        raise NotImplementedError("valid should be implemented.")

    def setup_model(self):
        raise NotImplementedError("setup_model should be implemented.")

    def setup_dataloader(self):
        raise NotImplementedError("setup_dataloader should be implemented.")

