from utils import *
from modules import *
import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
from parakeet.g2p.text.symbols import symbols
from parakeet.modules.utils import *
from parakeet.modules.post_convnet import PostConvNet

class Encoder(dg.Layer):
    def __init__(self,
                 n_src_vocab,
                 len_max_seq,
                 d_word_vec,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 d_model,
                 d_inner,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout=0.1):
        super(Encoder, self).__init__()
        n_position = len_max_seq + 1

        self.src_word_emb = dg.Embedding(size=[n_src_vocab, d_word_vec], padding_idx=0)
        self.pos_inp = get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0)
        self.position_enc = dg.Embedding(size=[n_position, d_word_vec],
                                 padding_idx=0,
                                 param_attr=fluid.ParamAttr(
                                     initializer=fluid.initializer.NumpyArrayInitializer(self.pos_inp),
                                     trainable=False))
        self.layer_stack = [FFTBlock(d_model, d_inner, n_head, d_k, d_v, fft_conv1d_kernel, fft_conv1d_padding, dropout=dropout) for _ in range(n_layers)]
        for i, layer in enumerate(self.layer_stack):
            self.add_sublayer('fft_{}'.format(i), layer)

    def forward(self, character, text_pos):
        enc_slf_attn_list = []
        # -- prepare masks
        # shape character (N, T)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=character, seq_q=character)
        non_pad_mask = get_non_pad_mask(character)

        # -- Forward
        enc_output = self.src_word_emb(character) + self.position_enc(text_pos) #(N, T, C)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            enc_slf_attn_list += [enc_slf_attn]
        
        return enc_output, non_pad_mask, enc_slf_attn_list

class Decoder(dg.Layer):
    def __init__(self,
                 len_max_seq,
                 d_word_vec,
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
        self.pos_inp = get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0)
        self.position_enc = dg.Embedding(size=[n_position, d_word_vec],
                                 padding_idx=0,
                                 param_attr=fluid.ParamAttr(
                                     initializer=fluid.initializer.NumpyArrayInitializer(self.pos_inp),
                                     trainable=False))
        self.layer_stack = [FFTBlock(d_model, d_inner, n_head, d_k, d_v, fft_conv1d_kernel, fft_conv1d_padding, dropout=dropout) for _ in range(n_layers)] 
        for i, layer in enumerate(self.layer_stack):
            self.add_sublayer('fft_{}'.format(i), layer)
    
    def forward(self, enc_seq, enc_pos):
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

class FastSpeech(dg.Layer):
    def __init__(self, cfg):
        " FastSpeech"
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(n_src_vocab=len(symbols)+1,
                               len_max_seq=cfg.max_sep_len,
                               d_word_vec=cfg.embedding_size,
                               n_layers=cfg.encoder_n_layer,
                               n_head=cfg.encoder_head,
                               d_k=64,
                               d_v=64,
                               d_model=cfg.hidden_size,
                               d_inner=cfg.encoder_conv1d_filter_size,
                               fft_conv1d_kernel=cfg.fft_conv1d_filter, 
                               fft_conv1d_padding=cfg.fft_conv1d_padding,
                               dropout=0.1)
        self.length_regulator = LengthRegulator(input_size=cfg.hidden_size, 
                                                out_channels=cfg.duration_predictor_output_size, 
                                                filter_size=cfg.duration_predictor_filter_size, 
                                                dropout=cfg.dropout)
        self.decoder = Decoder(len_max_seq=cfg.max_sep_len,
                                d_word_vec=cfg.embedding_size,
                                n_layers=cfg.decoder_n_layer,
                                n_head=cfg.decoder_head,
                                d_k=64,
                                d_v=64,
                                d_model=cfg.hidden_size,
                                d_inner=cfg.decoder_conv1d_filter_size,
                                fft_conv1d_kernel=cfg.fft_conv1d_filter, 
                                fft_conv1d_padding=cfg.fft_conv1d_padding,
                                dropout=0.1)
        self.mel_linear = dg.Linear(cfg.decoder_output_size, cfg.audio.num_mels)
        self.postnet = PostConvNet(n_mels=80,
                 num_hidden=512,
                 filter_size=5,
                 padding=int(5 / 2),
                 num_conv=5,
                 outputs_per_step=1,
                 use_cudnn=True,
                 dropout=0.1)

    def forward(self, character, text_pos, mel_pos=None, length_target=None, alpha=1.0):
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
            decoder_output = self.decoder(length_regulator_output, decoder_pos)

            mel_output = self.mel_linear(decoder_output)
            mel_output_postnet = self.postnet(mel_output) + mel_output

            return mel_output, mel_output_postnet