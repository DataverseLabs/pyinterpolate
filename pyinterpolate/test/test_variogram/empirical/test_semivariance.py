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

REFERENCE_INPUT_WEIGHTED = np.array(
    [
        [
            [0, 0, 735],
            [0, 1, 45],
            [0, 2, 125],
            [0, 3, 167],
            [1, 0, 450],
            [1, 1, 337],
            [1, 2, 95],
            [1, 3, 245],
            [2, 0, 124],
            [2, 1, 430],
            [2, 2, 230],
            [2, 3, 460],
            [3, 0, 75],
            [3, 1, 20],
            [3, 2, 32],
            [3, 3, 20]
        ],
        [
            [0, 0, 2],
            [0, 1, 3],
            [0, 2, 2],
            [0, 3, 3],
            [1, 0, 1],
            [1, 1, 3],
            [1, 2, 3],
            [1, 3, 2],
            [2, 0, 1],
            [2, 1, 2],
            [2, 2, 3],
            [2, 3, 1],
            [3, 0, 2],
            [3, 1, 2],
            [3, 2, 2],
            [3, 3, 1]
        ]]
)

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

EXPECTED_OUTPUT_WEIGHTED = np.array([
    [1, 30651.4, 48],
    [2, 40098.6, 68]
])

EXPECTED_OUTPUT_WEIGHTED_DIR = np.array([
    [2, 34480.6, 18],
    [4, 16409.3, 8],
    [6, 4166.9, 2]
])

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

    # WEIGHTED CASES

    def test_calculate_weighted_omnidirectional(self):
        _input = REFERENCE_INPUT_WEIGHTED.copy()
        arr = _input[0]
        weight_arr = _input[1]
        smv = calculate_semivariance(arr, 1, 3, weights=weight_arr[:, -1])
        arr_check = np.allclose(smv, EXPECTED_OUTPUT_WEIGHTED, rtol=0.1)
        err_msg = f'Given arrays are not equal. Expected output is {EXPECTED_OUTPUT_WEIGHTED} ' \
                  f'and calculated output is {smv}'
        self.assertTrue(arr_check, err_msg)

    def test_calculate_directional_weighted(self):
        _input = REFERENCE_INPUT_WEIGHTED.copy()
        arr = _input[0]
        weight_arr = _input[1]
        smv = calculate_semivariance(arr, 2, 7, weights=weight_arr[:, -1], direction=45, tolerance=0.01)
        arr_check = np.allclose(smv, EXPECTED_OUTPUT_WEIGHTED_DIR, rtol=0.1)
        err_msg = f'Given arrays are not equal. Expected output for directional weighted semivariogram is ' \
                  f'{EXPECTED_OUTPUT_WEIGHTED_DIR} and calculated output is {smv}'
        self.assertTrue(arr_check, err_msg)


if __name__ == '__main__':
    unittest.main()
