import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
from parakeet.modules.utils import *
from parakeet.models.fastspeech.FFTBlock import FFTBlock

class Decoder(dg.Layer):
    def __init__(self,
                 len_max_seq,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 d_model,
                 d_inner,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout=0.1):
        super(Decoder, self).__init__()

        n_position = len_max_seq + 1
        self.pos_inp = get_sinusoid_encoding_table(n_position, d_model, padding_idx=0)
        self.position_enc = dg.Embedding(size=[n_position, d_model],
                                 padding_idx=0,
                                 param_attr=fluid.ParamAttr(
                                     initializer=fluid.initializer.NumpyArrayInitializer(self.pos_inp),
                                     trainable=False))
        self.layer_stack = [FFTBlock(d_model, d_inner, n_head, d_k, d_v, fft_conv1d_kernel, fft_conv1d_padding, dropout=dropout) for _ in range(n_layers)] 
        for i, layer in enumerate(self.layer_stack):
            self.add_sublayer('fft_{}'.format(i), layer)
    
    def forward(self, enc_seq, enc_pos):
        """
        Decoder layer of FastSpeech.
        
        Args:
            enc_seq (Variable), Shape(B, text_T, C), dtype: float32. 
                The output of length regulator.
            enc_pos (Variable, optional): Shape(B, T_mel),
                dtype: int64. The spectrum position. T_mel means the timesteps of input spectrum.
        Returns:
            dec_output (Variable), Shape(B, mel_T, C), the decoder output.
            dec_slf_attn_list (Variable), Shape(B, mel_T, mel_T), the decoder self attention list.
        """
        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            dec_slf_attn_list += [dec_slf_attn]

        return dec_output, dec_slf_attn_list