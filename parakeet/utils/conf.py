import attrdict
import flatdict
import argparse
from ruamel import yaml


class Config(attrdict.AttrDict):
    def dump(self, path):
        with open(path, 'wt') as f:
            yaml.safe_dump(dict(self), f, default_flow_style=False)

    def dumps(self):
        return yaml.safe_dump(dict(self), default_flow_style=False)

    @classmethod
    def from_file(cls, path):
        with open(path, 'rt') as f:
            c = yaml.safe_load(f)
        return cls(c)

    def merge_file(self, path):
        with open(path, 'rt') as f:
            other = yaml.safe_load(f)
        self.update(other)

    def merge_args(self, args):
        args_dict = vars(args)
        nested_dict = flatdict.FlatDict(args_dict, delimiter=".").as_dict()
        self.update(nested_dict)
        
    def merge(self, other):
        self.update(other)

    def flatten(self):
        flat = flatdict.FlatDict(self, delimiter='.')
        return flat

    def add_options_to_parser(self, parser):
        flat = self.flatten()
        g = parser.add_argument_group("config file options")
        for k, v in flat.items():
            g.add_argument("--{}".format(k), type=type(v), default=v, 
                           help="config file option: {}".format(k))