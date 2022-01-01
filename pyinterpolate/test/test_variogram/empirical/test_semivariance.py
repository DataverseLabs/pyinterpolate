import os
import unittest
import numpy as np
from pyinterpolate.variogram.empirical.semivariance import calculate_semivariance

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


class TestSemivariance(unittest.TestCase):

    # OMNIDIRECTIONAL CASES

    def test_calculate_semivariance_we_omni(self):
        output = calculate_semivariance(REFERENCE_INPUT_WE, step_size=STEP_SIZE, max_range=MAX_RANGE)
        are_close = np.allclose(output, EXPECTED_OUTPUT_WE_OMNI, rtol=1.e-3, atol=1.e-5)
        msg = 'There is a large mismatch between calculated semivariance and expected output.' \
              ' Omnidirectional semivariogram.'
        self.assertTrue(are_close, msg)

    def test_calculate_semivariance_zeros_omni(self):
        output = calculate_semivariance(REFERENCE_INPUT_ZEROS, step_size=STEP_SIZE, max_range=MAX_RANGE)
        mean_val = np.mean(output[:, 1])
        msg = 'Calculated semivariance should be equal to zero if we provide only zeros array.'
        self.assertEqual(mean_val, EXPECTED_OUTPUT_ZEROS, msg)

    def test_calculate_semivariance_omni(self):
        my_dir = os.path.dirname(__file__)
        filename = 'armstrong_data.npy'
        filepath = f'../../samples/point_data/numpy/{filename}'
        path_to_the_data = os.path.join(my_dir, filepath)
        arr = np.load(path_to_the_data)
        smv = calculate_semivariance(arr, 1, 2)
        lag1_test_value = smv[0][1]
        err_msg = f'Calculated semivariance for lag 1 should be equal to {EXPECTED_OUTPUT_ARMSTRONG_LAG1} for ' \
                  f'omnidirectional case and points within {filename} file.'
        self.assertAlmostEqual(lag1_test_value, EXPECTED_OUTPUT_ARMSTRONG_LAG1, places=1, msg=err_msg)

    # DIRECTIONAL CASES

    def test_calculate_semivariance_SN_lag1(self):
        my_dir = os.path.dirname(__file__)
        filename = 'armstrong_data.npy'
        filepath = f'../../samples/point_data/numpy/{filename}'
        path_to_the_data = os.path.join(my_dir, filepath)
        arr = np.load(path_to_the_data)
        smv = calculate_semivariance(arr, 1, 2, direction=0, tolerance=0.1)
        lag1_test_value = smv[0][1]
        err_msg = f'Calculated semivariance for lag 1 should be equal to {EXPECTED_OUTPUT_ARMSTRONG_NS_LAG1} for ' \
                  f'N-S direction case and points within {filename} file.'
        self.assertAlmostEqual(lag1_test_value, EXPECTED_OUTPUT_ARMSTRONG_NS_LAG1, places=2, msg=err_msg)

    def test_calculate_semivariance_WE_lag1(self):
        my_dir = os.path.dirname(__file__)
        filename = 'armstrong_data.npy'
        filepath = f'../../samples/point_data/numpy/{filename}'
        path_to_the_data = os.path.join(my_dir, filepath)
        arr = np.load(path_to_the_data)
        smv = calculate_semivariance(arr, 1, 2, direction=90, tolerance=0.1)
        lag1_test_value = smv[0][1]
        err_msg = f'Calculated semivariance for lag 1 should be equal to {EXPECTED_OUTPUT_ARMSTRONG_WE_LAG1} for ' \
                  f'W-E direction case and points within {filename} file.'
        self.assertAlmostEqual(lag1_test_value, EXPECTED_OUTPUT_ARMSTRONG_WE_LAG1, places=2, msg=err_msg)

    def test_calculate_semivariance_NW_SE_lag1(self):
        my_dir = os.path.dirname(__file__)
        filename = 'armstrong_data.npy'
        filepath = f'../../samples/point_data/numpy/{filename}'
        path_to_the_data = os.path.join(my_dir, filepath)
        arr = np.load(path_to_the_data)
        smv = calculate_semivariance(arr, 1, 4, direction=135, tolerance=0.01)
        lag1_test_value = smv[1][1]
        err_msg = f'Calculated semivariance for lag 1 should be equal to {EXPECTED_OUTPUT_ARMSTRONG_NW_SE_LAG2} for ' \
                  f'NW-SE direction case and points within {filename} file.'
        self.assertAlmostEqual(lag1_test_value, EXPECTED_OUTPUT_ARMSTRONG_NW_SE_LAG2, places=2, msg=err_msg)

    def test_calculate_semivariance_NE_SW_lag1(self):
        my_dir = os.path.dirname(__file__)
        filename = 'armstrong_data.npy'
        filepath = f'../../samples/point_data/numpy/{filename}'
        path_to_the_data = os.path.join(my_dir, filepath)
        arr = np.load(path_to_the_data)
        smv = calculate_semivariance(arr, 1, 3, direction=45, tolerance=0.01)
        lag1_test_value = smv[1][1]
        err_msg = f'Calculated semivariance for lag 1 should be equal to {EXPECTED_OUTPUT_ARMSTRONG_NE_SW_LAG2} for ' \
                  f'NE-SW direction case and points within {filename} file.'
        self.assertAlmostEqual(lag1_test_value, EXPECTED_OUTPUT_ARMSTRONG_NE_SW_LAG2, places=2, msg=err_msg)
    #
    # # WEIGHTED CASES
    #
    # def test_calculate_weighted_omnidirectional(self):
    #     pass
    #
    # def test_calculate_directional_weighted(self):
    #     pass


if __name__ == '__main__':
    unittest.main()