import unittest
import numpy as np
import paddle

from parakeet.modules import positional_encoding as pe

def positional_encoding(start_index, length, size, dtype="float32"):
    if (size % 2 != 0):
        raise ValueError("size should be divisible by 2")
    channel = np.arange(0, size, 2, dtype=dtype)
    index = np.arange(start_index, start_index + length, 1, dtype=dtype)
    p = np.expand_dims(index, -1) / (10000 ** (channel / float(size)))
    encodings = np.concatenate([np.sin(p), np.cos(p)], axis=-1)
    return encodings

def scalable_positional_encoding(start_index, length, size, omega):
    dtype = omega.dtype
    index = np.arange(start_index, start_index + length, 1, dtype=dtype)
    channel = np.arange(0, size, 2, dtype=dtype)

    p = np.reshape(omega, omega.shape + (1, 1)) \
      * np.expand_dims(index, -1) \
      / (10000 ** (channel / float(size)))

    encodings = np.concatenate([np.sin(p), np.cos(p)], axis=-1)
    return encodings

class TestPositionEncoding(unittest.TestCase):
    def __init__(self, start=0, length=20, size=16, dtype="float64"):
        super(TestPositionEncoding, self).__init__("runTest")
        self.spec = (start, length, size, dtype)
    
    def test_equality(self):
        start, length, size, dtype = self.spec
        position_embed1 = positional_encoding(start, length, size, dtype)
        position_embed2 = pe.positional_encoding(start, length, size, dtype)
        np.testing.assert_allclose(position_embed2.numpy(), position_embed1)
        
    def runTest(self):
        paddle.disable_static(paddle.CPUPlace())
        self.test_equality()

class TestScalablePositionEncoding(unittest.TestCase):
    def __init__(self, start=0, length=20, size=16, dtype="float64"):
        super(TestScalablePositionEncoding, self).__init__("runTest")
        self.spec = (start, length, size, dtype)
    
    def test_equality(self):
        start, length, size, dtype = self.spec
        omega = np.random.uniform(1, 2, size=(4,)).astype(dtype)
        position_embed1 = scalable_positional_encoding(start, length, size, omega)
        position_embed2 = pe.scalable_positional_encoding(start, length, size, paddle.to_tensor(omega))
        np.testing.assert_allclose(position_embed2.numpy(), position_embed1)
        
    def runTest(self):
        paddle.disable_static(paddle.CPUPlace())
        self.test_equality()


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    suite.addTest(TestPositionEncoding(0, 20, 16, "float64"))
    suite.addTest(TestScalablePositionEncoding(0, 20, 16))
    return suite