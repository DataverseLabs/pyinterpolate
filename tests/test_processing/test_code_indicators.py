import unittest
import numpy as np

from pyinterpolate.processing.transform.transform import code_indicators


COORD = np.array([
    [999, -999, 5]
])
T_ARR = [1, 3, 5, 6, 7]
EXP = np.array([[999, -999, 0, 0, 1, 1, 1]])


class TestCodeIndicators(unittest.TestCase):

    def test_1(self):
        inds_arr = code_indicators(COORD, T_ARR)
        arrs_eq = np.array_equal(inds_arr, EXP)
        self.assertTrue(arrs_eq)
