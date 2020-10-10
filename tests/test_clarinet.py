import unittest
import numpy as np

import paddle
paddle.set_default_dtype("float64")
paddle.disable_static(paddle.CPUPlace())

from parakeet.models import clarinet
from parakeet.modules import stft

class TestParallelWaveNet(unittest.TestCase):
    def test_io(self):
        net = clarinet.ParallelWaveNet([8, 8, 8], [1, 1, 1], 16, 12, 2)
        x = paddle.randn([4, 6073])
        condition = paddle.randn([4, 12, 6073])
        z, out_mu, out_log_std = net(x, condition)
        self.assertTupleEqual(z.numpy().shape, (4, 6073))
        self.assertTupleEqual(out_mu.numpy().shape, (4, 6073))
        self.assertTupleEqual(out_log_std.numpy().shape, (4, 6073))
        

class TestClariNet(unittest.TestCase):
    def setUp(self):
        encoder = clarinet.UpsampleNet([2, 2])
        teacher = clarinet.WaveNet(8, 3, 16, 3, 12, 2, "mog", -9.0)
        student = clarinet.ParallelWaveNet([8, 8, 8, 8, 8, 8], [1, 1, 1, 1, 1, 1], 16, 12, 2)
        stft_module = stft.STFT(16, 4, 8)
        net = clarinet.Clarinet(encoder, teacher, student, stft_module, -6.0, lmd=4)
        print("context size is: ", teacher.context_size)
        self.net = net
        
    def test_io(self):
        audio = paddle.randn([4, 1366])
        mel = paddle.randn([4, 12, 512]) # 512 * 4 =2048
        audio_start = paddle.zeros([4], dtype="int64")
        loss = self.net(audio, mel, audio_start, clip_kl=True)
        loss["loss"].numpy()
        
    def test_synthesis(self):
        mel = paddle.randn([4, 12, 512]) # 64 = 246 / 4
        out = self.net.synthesis(mel)
        self.assertTupleEqual(out.numpy().shape, (4, 2048))
        