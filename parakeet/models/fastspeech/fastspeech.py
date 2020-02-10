import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
from parakeet.g2p.text.symbols import symbols
from parakeet.modules.utils import *
from parakeet.modules.post_convnet import PostConvNet
from parakeet.modules.layers import Linear
from parakeet.models.fastspeech.utils import *
from parakeet.models.fastspeech.LengthRegulator import LengthRegulator
from parakeet.models.fastspeech.encoder import Encoder
from parakeet.models.fastspeech.decoder import Decoder

class FastSpeech(dg.Layer):
    def __init__(self, cfg):
        " FastSpeech"
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(n_src_vocab=len(symbols)+1,
                               len_max_seq=cfg.max_sep_len,
                               n_layers=cfg.encoder_n_layer,
                               n_head=cfg.encoder_head,
                               d_k=cfg.fs_hidden_size // cfg.encoder_head,
                               d_v=cfg.fs_hidden_size // cfg.encoder_head,
                               d_model=cfg.fs_hidden_size,
                               d_inner=cfg.encoder_conv1d_filter_size,
                               fft_conv1d_kernel=cfg.fft_conv1d_filter, 
                               fft_conv1d_padding=cfg.fft_conv1d_padding,
                               dropout=0.1)
        self.length_regulator = LengthRegulator(input_size=cfg.fs_hidden_size, 
                                                out_channels=cfg.duration_predictor_output_size, 
                                                filter_size=cfg.duration_predictor_filter_size, 
                                                dropout=cfg.dropout)
        self.decoder = Decoder(len_max_seq=cfg.max_sep_len,
                                n_layers=cfg.decoder_n_layer,
                                n_head=cfg.decoder_head,
                                d_k=cfg.fs_hidden_size // cfg.decoder_head,
                                d_v=cfg.fs_hidden_size // cfg.decoder_head,
                                d_model=cfg.fs_hidden_size,
                                d_inner=cfg.decoder_conv1d_filter_size,
                                fft_conv1d_kernel=cfg.fft_conv1d_filter, 
                                fft_conv1d_padding=cfg.fft_conv1d_padding,
                                dropout=0.1)
        self.mel_linear = Linear(cfg.fs_hidden_size, cfg.audio.num_mels * cfg.audio.outputs_per_step)
        self.postnet = PostConvNet(n_mels=cfg.audio.num_mels,
                 num_hidden=512,
                 filter_size=5,
                 padding=int(5 / 2),
                 num_conv=5,
                 outputs_per_step=cfg.audio.outputs_per_step,
                 use_cudnn=True,
                 dropout=0.1,
                 batchnorm_last=True)

    def forward(self, character, text_pos, mel_pos=None, length_target=None, alpha=1.0):
        """
        FastSpeech model.
        
        Args:
            character (Variable): Shape(B, T_text), dtype: float32. The input text
                characters. T_text means the timesteps of input characters.
            text_pos (Variable): Shape(B, T_text), dtype: int64. The input text
                position. T_text means the timesteps of input characters.
            mel_pos (Variable, optional): Shape(B, T_mel),
                dtype: int64. The spectrum position. T_mel means the timesteps of input spectrum.
            length_target (Variable, optional): Shape(B, T_text),
                dtype: int64. The duration of phoneme compute from pretrained transformerTTS.
            alpha (Constant): 
                dtype: float32. The hyperparameter to determine the length of the expanded sequence 
                mel, thereby controlling the voice speed.

        Returns:
            mel_output (Variable), Shape(B, mel_T, C), the mel output before postnet.
            mel_output_postnet (Variable), Shape(B, mel_T, C), the mel output after postnet.
            duration_predictor_output (Variable), Shape(B, text_T), the duration of phoneme compute 
            with duration predictor.
            enc_slf_attn_list (Variable), Shape(B, text_T, text_T), the encoder self attention list.
            dec_slf_attn_list (Variable), Shape(B, mel_T, mel_T), the decoder self attention list.
        """

        encoder_output, non_pad_mask, enc_slf_attn_list = self.encoder(character, text_pos)
        if fluid.framework._dygraph_tracer()._train_mode:
            
            length_regulator_output, duration_predictor_output = self.length_regulator(encoder_output,
                                                                                       target=length_target,
                                                                                       alpha=alpha)
            decoder_output, dec_slf_attn_list = self.decoder(length_regulator_output, mel_pos)

            mel_output = self.mel_linear(decoder_output)
            mel_output_postnet = self.postnet(mel_output) + mel_output

            return mel_output, mel_output_postnet, duration_predictor_output, enc_slf_attn_list, dec_slf_attn_list
        else:
            length_regulator_output, decoder_pos = self.length_regulator(encoder_output, alpha=alpha)
            decoder_output, _ = self.decoder(length_regulator_output, decoder_pos)
            mel_output = self.mel_linear(decoder_output)
            mel_output_postnet = self.postnet(mel_output) + mel_output

            return mel_output, mel_output_postnet