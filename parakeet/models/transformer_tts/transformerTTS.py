import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
from parakeet.models.transformer_tts.encoder import Encoder
from parakeet.models.transformer_tts.decoder import Decoder

class TransformerTTS(dg.Layer):
    def __init__(self, config):
        super(TransformerTTS, self).__init__()
        self.encoder = Encoder(config['embedding_size'], config['hidden_size'])
        self.decoder = Decoder(config['hidden_size'], config)
        self.config = config

    def forward(self, characters, mel_input, pos_text, pos_mel):
        # key (batch_size, seq_len, channel)
        # c_mask (batch_size, seq_len)
        # attns_enc (channel / 2, seq_len, seq_len)

        key, c_mask, attns_enc = self.encoder(characters, pos_text)
        
        # mel_output/postnet_output (batch_size, mel_len, n_mel)
        # attn_probs (128, mel_len, seq_len)
        # stop_preds (batch_size, mel_len, 1)
        # attns_dec (128, mel_len, mel_len)
        mel_output, postnet_output, attn_probs, stop_preds, attns_dec = self.decoder(key, key, mel_input, c_mask, pos_mel)

        return mel_output, postnet_output, attn_probs, stop_preds, attns_enc, attns_dec


    


            
            
