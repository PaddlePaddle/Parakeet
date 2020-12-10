from yacs.config import CfgNode 

_C = CfgNode(
    dict(
        valid_interval=1000, # validation
        save_interval=10000, # checkpoint
        max_iteration=900000, # max iteration to train
    )
)

def get_default_training_config():
    return _C.clone()
