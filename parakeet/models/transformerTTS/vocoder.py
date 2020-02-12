import paddle.fluid.dygraph as dg
import paddle.fluid as fluid
from parakeet.modules.customized import Conv1D
from parakeet.modules.utils import *
from parakeet.models.transformerTTS.CBHG import CBHG

class Vocoder(dg.Layer):
    """
    CBHG Network (mel -> linear)
    """
    def __init__(self, config):
        super(Vocoder, self).__init__()
        self.pre_proj = Conv1D(num_channels = config.audio.num_mels, 
                             num_filters = config.hidden_size,
                             filter_size=1)
        self.cbhg = CBHG(config.hidden_size, config.batch_size)
        self.post_proj = Conv1D(num_channels = config.hidden_size, 
                             num_filters = (config.audio.n_fft // 2) + 1,
                             filter_size=1)

    def forward(self, mel):
        mel = layers.transpose(mel, [0,2,1])
        mel = self.pre_proj(mel)
        mel = self.cbhg(mel)
        mag_pred = self.post_proj(mel)
        mag_pred = layers.transpose(mag_pred, [0,2,1])
        return mag_pred
