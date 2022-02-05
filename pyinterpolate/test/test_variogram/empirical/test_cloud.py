import os
import unittest
import numpy as np
from pyinterpolate.variogram.empirical.cloud import get_variogram_point_cloud

# REFERENCE INPUTS
REFERENCE_INPUT_WE = np.array([
    [0, 0, 8],
    [1, 0, 6],
    [2, 0, 4],
    [3, 0, 3],
    [4, 0, 6],
    [5, 0, 5],
    [6, 0, 7],
    [7, 0, 2],
    [8, 0, 8],
    [9, 0, 9],
    [10, 0, 5],
    [11, 0, 6],
    [12, 0, 3]
])
REFERENCE_INPUT_ZEROS = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 0],
    [2, 1, 0],
    [1, 2, 0],
    [2, 2, 0],
    [3, 3, 0],
    [3, 1, 0],
    [3, 2, 0]
])

# EXPECTED OUTPUTS
EXPECTED_OUTPUT_WE_OMNI = np.array([
    [1, 4.625, 24],
    [2, 5.227, 22],
    [3, 6.0, 20],
    [4, 4.444, 18],
    [5, 3.125, 16]
])

EXPECTED_OUTPUT_ZEROS = 0

EXPECTED_OUTPUT_ARMSTRONG_WE_LAG1 = 6.41
EXPECTED_OUTPUT_ARMSTRONG_NS_LAG1 = 4.98
EXPECTED_OUTPUT_ARMSTRONG_NE_SW_LAG2 = 7.459
EXPECTED_OUTPUT_ARMSTRONG_NW_SE_LAG2 = 7.806
EXPECTED_OUTPUT_ARMSTRONG_LAG1 = 5.69

# CONSTS
STEP_SIZE = 1
MAX_RANGE = 6


class TestVariogramPointCloud(unittest.TestCase):

    # OMNIDIRECTIONAL CASES
    def test_instance(self):
        cloud = get_variogram_point_cloud(
            REFERENCE_INPUT_WE,
            step_size=STEP_SIZE,
            max_range=MAX_RANGE
        )
        self.assertIsInstance(cloud, dict)

    def test_output_we_omni(self):
        cloud = get_variogram_point_cloud(
            REFERENCE_INPUT_WE,
            step_size=STEP_SIZE,
            max_range=MAX_RANGE
        )
        arr = []
        for lag in cloud.keys():
            vals = cloud[lag].copy()
            lvals = len(vals)
            smv = np.mean(vals) / 2
            arr.append([
                lag,
                smv,
                lvals
            ])
        are_close = np.allclose(arr, EXPECTED_OUTPUT_WE_OMNI, rtol=1.e-3, atol=1.e-5)
        msg = 'There is a large mismatch between calculated semivariance and expected output.' \
              ' Omnidirectional semivariogram.'
        self.assertTrue(are_close, msg)

    def test_zeros(self):
        cloud = get_variogram_point_cloud(
            REFERENCE_INPUT_ZEROS,
            step_size=STEP_SIZE,
            max_range=MAX_RANGE
        )
        msg = 'For array with zeros you should always obtain variance between points equal to zero!'
        for lag, smv_arr in cloud.items():
            mval = np.mean(smv_arr)
            self.assertEqual(mval, EXPECTED_OUTPUT_ZEROS, msg=msg)

    # DIRECTIONAL CASES
    

if __name__ == '__main__':
    unittest.main()
