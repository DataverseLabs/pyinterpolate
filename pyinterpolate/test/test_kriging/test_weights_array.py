import unittest

import numpy as np

from pyinterpolate.kriging.models.block.weight import weights_array


class TestWeightingArray(unittest.TestCase):

    def test_weighting_method(self):
        blocks = np.array([1, 2, 3, 4, 5])
        points = np.array([10, 20, 10, 20, 10])

        shape = (5, 5)

        weights = weights_array(shape, blocks, points)
        expected_arr = np.zeros(shape=shape)
        np.fill_diagonal(expected_arr, 3)
        self.assertTrue(np.array_equal(weights, expected_arr))
