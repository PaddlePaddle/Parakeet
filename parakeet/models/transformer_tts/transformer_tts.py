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

    def forward(self, characters, mel_input, pos_text, pos_mel, dec_slf_mask, enc_slf_mask=None, enc_query_mask=None, enc_dec_mask=None, dec_query_slf_mask=None, dec_query_mask=None):
        key, attns_enc = self.encoder(characters, pos_text, mask=enc_slf_mask, query_mask=enc_query_mask)   
       
        mel_output, postnet_output, attn_probs, stop_preds, attns_dec = self.decoder(key, key, mel_input, pos_mel, 
                                                                        mask=dec_slf_mask, zero_mask=enc_dec_mask, 
                                                                        m_self_mask=dec_query_slf_mask, m_mask=dec_query_mask )
        return mel_output, postnet_output, attn_probs, stop_preds, attns_enc, attns_dec

    


            
            
