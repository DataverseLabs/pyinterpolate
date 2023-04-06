import unittest
import numpy as np

from pyinterpolate.variogram.empirical.semivariance import create_triangles_mask


class TestCreateTrianglesMask(unittest.TestCase):

    def test_create_triangles_mask(self):
        bool1 = np.array([True, True, False, False, False])
        bool2 = np.array([False, True, True, False, False])

        expected_output = np.array([False, False, True, False, False])
        mask = create_triangles_mask(bool1, bool2)
        mask_test = np.array_equal(expected_output, mask)
        self.assertTrue(mask_test)
