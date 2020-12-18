from yacs.config import CfgNode as CN

_C = CN()
_C.data = CN(
    dict(
        batch_size=8, # batch size
        valid_size=16, # the first N examples are reserved for validation
        sample_rate=22050, # Hz, sample rate
        n_fft=2048, # fft frame size
        win_length=1024, # window size
        hop_length=256,  # hop size between ajacent frame
        # f_max=8000, # Hz, max frequency when converting to mel
        n_mels=80,  # mel bands
        train_clip_seconds=0.5, # audio clip length(in seconds)
    )
)

_C.model = CN(
    dict(
        upsample_factors=[16, 16],
        n_stack=3,
        n_loop=10,
        filter_size=2,
        residual_channels=128, # resiaudal channel in each flow
        loss_type="mog",
        output_dim=3, # single gaussian
        log_scale_min=-9.0,
    )
)

_C.training = CN(
    dict(
        lr=1e-3, # learning rates
        anneal_rate=0.5, # learning rate decay rate
        anneal_interval=200000, # decrese lr by annel_rate every anneal_interval steps
        valid_interval=1000, # validation
        save_interval=10000, # checkpoint
        max_iteration=3000000, # max iteration to train
        gradient_max_norm=100.0 # global norm of gradients
    )
)

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
