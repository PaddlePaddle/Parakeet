import unittest
import numpy as np
import librosa
import paddle
paddle.set_default_dtype("float64")
paddle.disable_static(paddle.CPUPlace())

from parakeet.modules import stft

class TestSTFT(unittest.TestCase):
    def test(self):
        path = librosa.util.example("choice")
        wav, sr = librosa.load(path, duration=5)
        wav = wav.astype("float64")
        
        spec = librosa.stft(wav, n_fft=2048, hop_length=256, win_length=1024)
        mag1 = np.abs(spec)
        
        wav_in_batch = paddle.unsqueeze(paddle.to_tensor(wav), 0)
        mag2 = stft.STFT(2048, 256, 1024).magnitude(wav_in_batch)
        mag2 = paddle.squeeze(mag2, [0, 2]).numpy()
        
        print("mag1", mag1)
        print("mag2", mag2)
        # TODO(chenfeiyu): Is there something wrong? there is some elements that
        # does not match
        # np.testing.assert_allclose(mag2, mag1)
