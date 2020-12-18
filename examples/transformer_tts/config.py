from yacs.config import CfgNode as CN

_C = CN()
_C.data = CN(
    dict(
        batch_size=16, # batch size
        valid_size=64, # the first N examples are reserved for validation
        sample_rate=22050, # Hz, sample rate
        n_fft=1024, # fft frame size
        win_length=1024, # window size
        hop_length=256,  # hop size between ajacent frame
        f_max=8000, # Hz, max frequency when converting to mel
        d_mel=80,  # mel bands
        padding_idx=0, # text embedding's padding index
        mel_start_value=0.5, # value for starting frame
        mel_end_value=-0.5, # # value for ending frame
    )
)

_C.model = CN(
    dict(
        d_encoder=512,  # embedding & encoder's internal size
        d_decoder=256,  # decoder's internal size
        n_heads=4,  # actually it can differ at each layer
        d_ffn=1024,  # encoder_d_ffn & decoder_d_ffn
        encoder_layers=4,  # number of transformer encoder layer
        decoder_layers=4,  # number of transformer decoder layer
        d_prenet=256,  # decprenet's hidden size (d_mel=>d_prenet=>d_decoder)
        d_postnet=256,  # decoder postnet(cnn)'s internal channel
        postnet_layers=5,  # decoder postnet(cnn)'s layer
        postnet_kernel_size=5,  # decoder postnet(cnn)'s kernel size
        max_reduction_factor=10,  # max_reduction factor
        dropout=0.1,  # global droput probability
        stop_loss_scale=8.0, # scaler for stop _loss
        decoder_prenet_dropout=0.5, # decoder prenet dropout probability
    )
)

_C.training = CN(
    dict(
        lr=1e-4, # learning rate
        drop_n_heads=[[0, 0], [15000, 1]],
        reduction_factor=[[0, 10], [80000, 4], [200000, 2]],
        plot_interval=1000, # plot attention and spectrogram
        valid_interval=1000, # validation
        save_interval=10000, # checkpoint
        max_iteration=900000, # max iteration to train
    )
)

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
