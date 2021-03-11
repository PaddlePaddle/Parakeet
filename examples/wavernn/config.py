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

from yacs.config import CfgNode as CN

_C = CN()

_C.data = CN(
    dict(
        sample_rate=22050,                  
        n_fft=2048,                         # fft frame size
        fft_bins=1025,                      # fft bins
        num_mels=80,                        # mel bands
        hop_length=275,                     # hop size between ajacent frame
        win_length=1100,                    # window size
        fmin=40,                            # Hz, min frequency when converting to mel
        fmax=11025,                         # Hz, max frequency when converting to mel
        min_level_db=-100,                  # min level db
        ref_level_db=20,                    # res level db
        bits=9,                             # bit depth of signal
        mu_law=True,                        # Recommended to suppress noise if using raw bits
        peak_norm=True,                     # Normalise to the peak of each wav file
        valid_size=50,                      # How many unseen samples to put aside for testing
        batch_size=48                       # batch size when training
    )
)

_C.model = CN(
    dict(
        mode='RAW',                         # either 'RAW'(softmax on raw bits) or 'MOL' (sample from mixture of logistics)
        upsample_factors=[5, 5, 11],        # this needs to correctly factorise hop_length
        rnn_dims=512,                       # thie hidden dim of rnn
        fc_dims=512,
        compute_dims=128,
        res_out_dims=128,
        res_blocks=10,
        gen_batched=True,                   # whether to genenate sample in batch mode
        target=11000,                       # target number of samples to be generated in each batch entry
        overlap=550,                        # number of samples for crossfading between batches
        pad=2                               # this will pad the input so that the resnet can 'see' wider than input length
    )
)

_C.training = CN(
    dict(
        lr=1e-4,
        save_interval=25000,                # the iteration interval of saving checkpoint
        generate_at_checkpoint=5,           # number of samples to generate at each checkpoint
        max_iteration=10000000,             # total number of training steps
        seq_len=275*15,                     # the length of sequence when training, seq_len can be adjusted to increase training sequence length (will increase GPU usage)
        clip_grad_norm=4,                   # clip grad norm
        valid_interval=1000,                # the iteration interval of validating
        valid_generate_valid_interval=25000 # the iteration interval of generating valid samples
    )
)


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

if __name__ == '__main__':
    config = get_cfg_defaults()
    config.data.mode = config.model.mode
    config.freeze()
    print(config.data)
