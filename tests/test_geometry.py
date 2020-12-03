import unittest
import numpy as np

import paddle
paddle.set_default_dtype("float64")
paddle.disable_static(paddle.CPUPlace())

from parakeet.modules import geometry as geo

class TestShuffleDim(unittest.TestCase):
    def test_perm(self):
        x = paddle.randn([2, 3, 4, 6])
        y = geo.shuffle_dim(x, 2, [3, 2, 1, 0])
        np.testing.assert_allclose(x.numpy()[0, 0, :, 0], y.numpy()[0, 0, ::-1, 0])
        
    def test_random_perm(self):
        x = paddle.randn([2, 3, 4, 6])
        y = geo.shuffle_dim(x, 2)
        np.testing.assert_allclose(x.numpy().sum(2), y.numpy().sum(2))