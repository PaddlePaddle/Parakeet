import numpy as np
import math
import utils
import paddle.fluid.dygraph as dg
import paddle.fluid.layers as layers
import paddle.fluid as fluid
from parakeet.modules.layers import Conv1D
from parakeet.modules.multihead_attention import MultiheadAttention
from parakeet.modules.feed_forward import PositionwiseFeedForward



class FFTBlock(dg.Layer):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, filter_size, padding, dropout=0.2):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiheadAttention(d_model, d_k, d_v, num_head=n_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, filter_size =filter_size, padding =padding, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        """
        Feed Forward Transformer block in FastSpeech.
        
        Args:
            enc_input (Variable): Shape(B, T, C), dtype: float32. The embedding characters input. 
                T means the timesteps of input.
            non_pad_mask (Variable): Shape(B, T, 1), dtype: int64. The mask of sequence.
            slf_attn_mask (Variable): Shape(B, len_q, len_k), dtype: int64. The mask of self attention. 
                len_q means the sequence length of query, len_k means the sequence length of key.

        Returns:
            output (Variable), Shape(B, T, C), the output after self-attention & ffn.
            slf_attn (Variable), Shape(B * n_head, T, T), the self attention.
        """
        output, slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        output *= non_pad_mask

        output = self.pos_ffn(output)
        output *= non_pad_mask

        return output, slf_attn


class LengthRegulator(dg.Layer):
    def __init__(self, input_size, out_channels, filter_size, dropout=0.1):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = DurationPredictor(input_size=input_size, 
                                                    out_channels=out_channels, 
                                                    filter_size=filter_size, 
                                                    dropout=dropout)

    def LR(self, x, duration_predictor_output, alpha=1.0):
        output = []
        batch_size = x.shape[0]
        for i in range(batch_size):
            output.append(self.expand(x[i:i+1], duration_predictor_output[i:i+1], alpha))
        output = self.pad(output)
        return output
    
    def pad(self, input_ele):
        max_len = max([input_ele[i].shape[0] for i in range(len(input_ele))])
        out_list = []
        for i in range(len(input_ele)):
            pad_len = max_len - input_ele[i].shape[0]
            one_batch_padded = layers.pad(
                input_ele[i], [0, pad_len, 0, 0], pad_value=0.0)
            out_list.append(one_batch_padded)
        out_padded = layers.stack(out_list)
        return out_padded
    
    def expand(self, batch, predicted, alpha):
        out = []
        time_steps = batch.shape[1]
        fertilities = predicted.numpy()
        batch = layers.squeeze(batch,[0]) 
        
        
        for i in range(time_steps):
            if fertilities[0,i]==0:
                continue
            out.append(layers.expand(batch[i: i + 1, :], [int(fertilities[0,i]), 1]))
        out = layers.concat(out, axis=0)
        return out
    

    def forward(self, x, alpha=1.0, target=None):
        """
        Length Regulator block in FastSpeech.
        
        Args:
            x (Variable): Shape(B, T, C), dtype: float32. The encoder output.
            alpha (Constant): dtype: float32. The hyperparameter to determine the length of 
                the expanded sequence mel, thereby controlling the voice speed.
            target (Variable): (Variable, optional): Shape(B, T_text),
                dtype: int64. The duration of phoneme compute from pretrained transformerTTS.

        Returns:
            output (Variable), Shape(B, T, C), the output after exppand.
            duration_predictor_output (Variable), Shape(B, T, C), the output of duration predictor.
        """
        duration_predictor_output = self.duration_predictor(x)
        if fluid.framework._dygraph_tracer()._train_mode:
            output = self.LR(x, target)
            return output, duration_predictor_output
        else:
            duration_predictor_output = layers.round(duration_predictor_output)
            output = self.LR(x, duration_predictor_output, alpha)
            mel_pos = dg.to_variable([i+1 for i in range(output.shape[1])])
            return output, mel_pos

class DurationPredictor(dg.Layer):
    def __init__(self, input_size, out_channels, filter_size, dropout=0.1):
        super(DurationPredictor, self).__init__()
        self.input_size = input_size
        self.out_channels = out_channels
        self.filter_size = filter_size
        self.dropout = dropout

        self.conv1 = Conv1D(in_channels = self.input_size, 
                        out_channels = self.out_channels, 
                        filter_size = self.filter_size,
                        padding=1,
                        data_format='NTC')
        self.conv2 = Conv1D(in_channels = self.out_channels, 
                        out_channels = self.out_channels, 
                        filter_size = self.filter_size,
                        padding=1,
                        data_format='NTC')
        self.layer_norm1 = dg.LayerNorm(self.out_channels)
        self.layer_norm2 = dg.LayerNorm(self.out_channels)

        self.linear =dg.Linear(self.out_channels, 1)

    def forward(self, encoder_output):
        """
        Duration Predictor block in FastSpeech.
        
        Args:
            encoder_output (Variable): Shape(B, T, C), dtype: float32. The encoder output.
        Returns:
            out (Variable), Shape(B, T, C), the output of duration predictor.
        """
        # encoder_output.shape(N, T, C)
        out = layers.dropout(layers.relu(self.layer_norm1(self.conv1(encoder_output))), self.dropout)
        out = layers.dropout(layers.relu(self.layer_norm2(self.conv2(out))), self.dropout)
        out = layers.relu(self.linear(out))
        out = layers.squeeze(out, axes=[-1])
            
        return out

        
