import os
import unittest
import numpy as np
from pyinterpolate.variogram.empirical.covariance import calculate_covariance

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

EXPECTED_OUTPUT_OMNI = np.array([
    [0, 4.248, 0, 0],
    [1, -0.543, 24, 30.71],
    [2, -0.795, 22, 30.25],
    [3, -1.26, 20, 31.36],
    [4, -0.197, 18, 30.864],
    [5, 1.234, 16, 28.891]
])

EXPECTED_VARIANCE_CO_OUTPUT = 4.248
EXPECTED_OUTPUT_ZEROS = 0

EXPECTED_OUTPUT_ARMSTRONG_WE_LAG1 = 4.643
EXPECTED_OUTPUT_ARMSTRONG_NS_LAG1 = 9.589
EXPECTED_OUTPUT_ARMSTRONG_NE_SW_LAG2 = 4.551
EXPECTED_OUTPUT_ARMSTRONG_NW_SE_LAG2 = 6.331
EXPECTED_OUTPUT_ARMSTRONG_LAG1 = 6.649

# CONSTS
STEP_SIZE = 1
MAX_RANGE = 6
RETURN_MEAN_SQUARED = True


class TestCovariance(unittest.TestCase):

    # OMNIDIRECTIONAL CASES

    def test_variance_c0(self):
        output = calculate_covariance(REFERENCE_INPUT_WE, step_size=STEP_SIZE, max_range=MAX_RANGE,
                                      return_lag_squared_means=RETURN_MEAN_SQUARED)
        c0 = output[0][1]
        msg = f'Expected variance of the input dataset is {EXPECTED_VARIANCE_CO_OUTPUT} but ' \
              f'{c0} was returned.'
        self.assertAlmostEqual(c0, EXPECTED_VARIANCE_CO_OUTPUT, 2, msg=msg)

    def test_calculate_covariance_single_row(self):
        output = calculate_covariance(REFERENCE_INPUT_WE, step_size=STEP_SIZE, max_range=MAX_RANGE,
                                      return_lag_squared_means=RETURN_MEAN_SQUARED)
        are_close = np.allclose(output, EXPECTED_OUTPUT_OMNI, rtol=1.e-2)
        msg = 'The difference between expected values and calculated covariances are too large, check calculations.'
        self.assertTrue(are_close, msg)

    def test_calculate_covariance_zeros_omni(self):
        output = calculate_covariance(REFERENCE_INPUT_ZEROS, step_size=STEP_SIZE, max_range=MAX_RANGE)
        mean_val = np.mean(output[:, 1])
        msg = 'Calculated covariance should be equal to zero if we provide only zeros array.'
        self.assertEqual(mean_val, EXPECTED_OUTPUT_ZEROS, msg)

    def test_calculate_covariance_omni(self):
        my_dir = os.path.dirname(__file__)
        filename = 'armstrong_data.npy'
        filepath = f'../../samples/point_data/numpy/{filename}'
        path_to_the_data = os.path.join(my_dir, filepath)
        arr = np.load(path_to_the_data)
        cov = calculate_covariance(arr, 1, 2, get_c0=False)
        lag1_test_value = cov[0][1]
        err_msg = f'Calculated covariance for lag 1 should be equal to {EXPECTED_OUTPUT_ARMSTRONG_LAG1} for ' \
                  f'omnidirectional case and points within {filename} file.'
        self.assertAlmostEqual(lag1_test_value, EXPECTED_OUTPUT_ARMSTRONG_LAG1, places=2, msg=err_msg)

    # DIRECTIONAL CASES

    def test_calculate_covariance_SN_lag1(self):
        my_dir = os.path.dirname(__file__)
        filename = 'armstrong_data.npy'
        filepath = f'../../samples/point_data/numpy/{filename}'
        path_to_the_data = os.path.join(my_dir, filepath)
        arr = np.load(path_to_the_data)
        cov = calculate_covariance(arr, 1, 2, direction=0, tolerance=0.01, get_c0=False)
        lag1_test_value = cov[0][1]
        err_msg = f'Calculated covariance for lag 1 should be equal to {EXPECTED_OUTPUT_ARMSTRONG_NS_LAG1} for ' \
                  f'N-S direction case and points within {filename} file.'
        self.assertAlmostEqual(lag1_test_value, EXPECTED_OUTPUT_ARMSTRONG_NS_LAG1, places=2, msg=err_msg)

    def test_calculate_semivariance_WE_lag1(self):
        my_dir = os.path.dirname(__file__)
        filename = 'armstrong_data.npy'
        filepath = f'../../samples/point_data/numpy/{filename}'
        path_to_the_data = os.path.join(my_dir, filepath)
        arr = np.load(path_to_the_data)
        cov = calculate_covariance(arr, 1, 2, direction=90, tolerance=0.1, get_c0=False)
        lag1_test_value = cov[0][1]
        err_msg = f'Calculated semivariance for lag 1 should be equal to {EXPECTED_OUTPUT_ARMSTRONG_WE_LAG1} for ' \
                  f'W-E direction case and points within {filename} file.'
        self.assertAlmostEqual(lag1_test_value, EXPECTED_OUTPUT_ARMSTRONG_WE_LAG1, places=2, msg=err_msg)

    def test_calculate_semivariance_NW_SE_lag1(self):
        my_dir = os.path.dirname(__file__)
        filename = 'armstrong_data.npy'
        filepath = f'../../samples/point_data/numpy/{filename}'
        path_to_the_data = os.path.join(my_dir, filepath)
        arr = np.load(path_to_the_data)
        cov = calculate_covariance(arr, 1, 4, direction=135, tolerance=0.01, get_c0=False)
        lag1_test_value = cov[1][1]
        err_msg = f'Calculated covariance for lag 1 should be equal to {EXPECTED_OUTPUT_ARMSTRONG_NW_SE_LAG2} for ' \
                  f'NW-SE direction case and points within {filename} file.'
        self.assertAlmostEqual(lag1_test_value, EXPECTED_OUTPUT_ARMSTRONG_NW_SE_LAG2, places=2, msg=err_msg)

    def test_calculate_semivariance_NE_SW_lag1(self):
        my_dir = os.path.dirname(__file__)
        filename = 'armstrong_data.npy'
        filepath = f'../../samples/point_data/numpy/{filename}'
        path_to_the_data = os.path.join(my_dir, filepath)
        arr = np.load(path_to_the_data)
        cov = calculate_covariance(arr, 1, 3, direction=45, tolerance=0.01, get_c0=False)
        lag1_test_value = cov[1][1]
        err_msg = f'Calculated coivariance for lag 1 should be equal to {EXPECTED_OUTPUT_ARMSTRONG_NE_SW_LAG2} for ' \
                  f'NE-SW direction case and points within {filename} file.'
        self.assertAlmostEqual(lag1_test_value, EXPECTED_OUTPUT_ARMSTRONG_NE_SW_LAG2, places=2, msg=err_msg)


if __name__ == '__main__':
    unittest.main()
