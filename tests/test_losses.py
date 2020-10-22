import unittest
import paddle
paddle.set_device("cpu")
import numpy as np

from parakeet.modules.losses import weighted_mean, masked_l1_loss, masked_softmax_with_cross_entropy

class TestWeightedMean(unittest.TestCase):
    def test(self):
        x = paddle.arange(0, 10, dtype="float64").unsqueeze(-1).broadcast_to([10, 3])
        mask = (paddle.arange(0, 10, dtype="float64") > 4).unsqueeze(-1)
        loss = weighted_mean(x, mask)
        self.assertAlmostEqual(loss.numpy()[0], 7)


class TestMaskedL1Loss(unittest.TestCase):
    def test(self):
        x = paddle.arange(0, 10, dtype="float64").unsqueeze(-1).broadcast_to([10, 3])
        y = paddle.zeros_like(x)
        mask = (paddle.arange(0, 10, dtype="float64") > 4).unsqueeze(-1)
        loss = masked_l1_loss(x, y, mask)
        print(loss)
        self.assertAlmostEqual(loss.numpy()[0], 7)


class TestMaskedCrossEntropy(unittest.TestCase):
    def test(self):
        x = paddle.randn([3, 30, 8], dtype="float64")
        y = paddle.randint(0, 8, [3, 30], dtype="int64").unsqueeze(-1) # mind this
        mask = paddle.fluid.layers.sequence_mask(
            paddle.to_tensor([30, 18, 27]), dtype="int64").unsqueeze(-1)
        loss = masked_softmax_with_cross_entropy(x, y, mask)
        print(loss)
