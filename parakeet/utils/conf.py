import attrdict
import flatdict
import argparse
import yaml


class Config(attrdict.AttrDict):
    def dump(self, path):
        with open(path, 'wt') as f:
            yaml.safe_dump(dict(self), f, default_flow_style=None)

    def dumps(self):
        return yaml.safe_dump(dict(self), default_flow_style=None)

    @classmethod
    def from_file(cls, path):
        with open(path, 'rt') as f:
            c = yaml.safe_load(f)
        return cls(c)

    def merge_file(self, path):
        with open(path, 'rt') as f:
            other = yaml.safe_load(f)
        self.update(self + other)

    def merge_args(self, args):
        args_dict = vars(args)
        args_dict.pop("config") # exclude config file path
        args_dict = {k: v for k, v in args_dict.items() if v is not None}
        nested_dict = flatdict.FlatDict(args_dict, delimiter=".").as_dict()
        self.update(self + nested_dict)
        
    def merge(self, other):
        self.update(self + other)

    def flatten(self):
        flat = flatdict.FlatDict(self, delimiter='.')
        return flat

    def add_options_to_parser(self, parser):
        parser.add_argument(
            "--config", type=str, 
            help="extra config file to override the default config")
        flat = self.flatten()
        g = parser.add_argument_group("config file options")
        for k, v in flat.items():
            g.add_argument("--{}".format(k), type=type(v), 
                           help="config file option: {}".format(k))