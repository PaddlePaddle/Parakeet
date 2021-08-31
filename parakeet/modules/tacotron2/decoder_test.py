# 对齐了
import paddle
import torch
import numpy
from decoder import Prenet as DecoderPrenet
from decoder_torch import Prenet as DecoderPrenet_torch

embed_dim= 512           # embedding dimension in encoder prenet
eprenet_conv_layers=0 # number of conv layers in encoder prenet
                        # if set to 0, no encoder prenet will be used
eprenet_conv_filts=0  # filter size of conv layers in encoder prenet
eprenet_conv_chans= 0  # number of channels of conv layers in encoder prenet
dprenet_layers=2      # number of layers in decoder prenet
dprenet_units=256     # number of units in decoder prenet
adim= 512              # attention dimension
aheads=8              # number of attention heads
elayers=6             # number of encoder layers
eunits=1024           # number of encoder ff units
dlayers=6             # number of decoder layers
dunits=1024           # number of decoder ff units
positionwise_layer_type='conv1d'  # type of position-wise layer
positionwise_conv_kernel_size=1 # kernel size of position wise conv layer
postnet_layers=5                # number of layers of postnset
postnet_filts=5                 # filter size of conv layers in postnet
postnet_chans=256               # number of channels of conv layers in postnet
use_masking=True                # whether to apply masking for padded part in loss calculation
bce_pos_weight=5.0              # weight of positive sample in binary cross entropy calculation
use_scaled_pos_enc=True         # whether to use scaled positional encoding
encoder_normalize_before=True   # whether to perform layer normalization before the input
decoder_normalize_before=True   # whether to perform layer normalization before the input
reduction_factor=1              # reduction factor
init_type='xavier_uniform'        # initialization type
init_enc_alpha=1.0              # initial value of alpha of encoder scaled position encoding
init_dec_alpha=1.0              # initial value of alpha of decoder scaled position encoding
eprenet_dropout_rate=0.0        # dropout rate for encoder prenet
dprenet_dropout_rate=0.5        # dropout rate for decoder prenet
postnet_dropout_rate=0.5        # dropout rate for postnet
transformer_enc_dropout_rate=0.1                # dropout rate for transformer encoder layer
transformer_enc_positional_dropout_rate=0.1     # dropout rate for transformer encoder positional encoding
transformer_enc_attn_dropout_rate=0.1           # dropout rate for transformer encoder attention layer
transformer_dec_dropout_rate=0.1                # dropout rate for transformer decoder layer
transformer_dec_positional_dropout_rate=0.1     # dropout rate for transformer decoder positional encoding
transformer_dec_attn_dropout_rate=0.1           # dropout rate for transformer decoder attention layer
transformer_enc_dec_attn_dropout_rate=0.1       # dropout rate for transformer encoder-decoder attention layer
use_guided_attn_loss=True                       # whether to use guided attention loss
num_heads_applied_guided_attn=2                 # number of layers to apply guided attention loss
num_layers_applied_guided_attn=2                # number of heads to apply guided attention loss
modules_applied_guided_attn=["encoder-decoder"] # modules to apply guided attention loss
guided_attn_loss_sigma=0.4                      # sigma in guided attention loss
guided_attn_loss_lambda=10.0                    # lambda in guided attention loss
use_batch_norm=True
padding_idx=0
idim = 174
odim = 80

decoder_prenet = DecoderPrenet(idim=odim,
            n_layers=dprenet_layers,
            n_units=dprenet_units,
            dropout_rate=dprenet_dropout_rate,)
decoder_prenet_torch= DecoderPrenet_torch(idim=odim,
            n_layers=dprenet_layers,
            n_units=dprenet_units,
            dropout_rate=dprenet_dropout_rate,)
            

np_query = numpy.random.rand(2,80)
query = paddle.to_tensor(np_query,dtype = 'float32')
query_torch = torch.tensor(np_query,dtype = torch.float32)
paddle_result = decoder_prenet(query)
print(paddle_result)
print(paddle_result.shape)
torch_result = decoder_prenet_torch(query_torch)
print("---------------")
print(torch_result)
print(torch_result.shape)