import unittest
import numpy as np
import paddle
paddle.set_default_dtype("float64")

from parakeet.modules import masking


def sequence_mask(lengths, max_length=None, dtype="bool"):
    max_length = max_length or np.max(lengths)
    ids = np.arange(max_length)
    return (ids < np.expand_dims(lengths, -1)).astype(dtype)

def future_mask(lengths, max_length=None, dtype="bool"):
    max_length = max_length or np.max(lengths)
    return np.tril(np.tril(np.ones(max_length))).astype(dtype)

class TestIDMask(unittest.TestCase):
    def test(self):
        ids = paddle.to_tensor(
            [[1, 2, 3, 0, 0, 0],
             [2, 4, 5, 6, 0, 0],
             [7, 8, 9, 0, 0, 0]]
        )
        mask = masking.id_mask(ids)
        self.assertTupleEqual(mask.numpy().shape, ids.numpy().shape)
        print(mask.numpy())
        
class TestFeatureMask(unittest.TestCase):
    def test(self):
        features = np.random.randn(3, 16, 8)
        lengths = [16, 14, 12]
        for i, length in enumerate(lengths):
            features[i, length:, :] = 0
        
        feature_tensor = paddle.to_tensor(features)
        mask = masking.feature_mask(feature_tensor, -1)
        self.assertTupleEqual(mask.numpy().shape, (3, 16, 1))
        print(mask.numpy().squeeze())
        
        
class TestCombineMask(unittest.TestCase):
    def test_bool_mask(self):
        lengths = np.array([12, 8, 9, 10])
        padding_mask = sequence_mask(lengths, dtype="bool")
        no_future_mask = future_mask(lengths, dtype="bool")
        combined_mask1 = np.expand_dims(padding_mask, 1) * no_future_mask
        
        print(paddle.to_tensor(padding_mask).dtype)
        print(paddle.to_tensor(no_future_mask).dtype)
        combined_mask2 = masking.combine_mask(
            paddle.to_tensor(padding_mask).unsqueeze(1), paddle.to_tensor(no_future_mask)
        )
        np.testing.assert_allclose(combined_mask2.numpy(), combined_mask1)
