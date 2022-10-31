import os
import unittest
import numpy as np
from pyinterpolate.variogram.empirical.semivariance import calculate_semivariance

# Imports for tests within this scope
from pyinterpolate.test.test_variogram.empirical.consts import get_armstrong_data, EmpiricalVariogramTestData,\
    EmpiricalSemivarianceData


gen_data = EmpiricalVariogramTestData()
sem_data = EmpiricalSemivarianceData()
armstrong_arr = get_armstrong_data()

class TestSemivariance(unittest.TestCase):

    # OMNIDIRECTIONAL CASES

    def test_calculate_semivariance_we_omni(self):
        arr = calculate_semivariance(gen_data.input_data_we,
                                     step_size=gen_data.param_step_size,
                                     max_range=gen_data.param_max_range)
        are_close = np.allclose(arr,
                                sem_data.output_we_omni,
                                rtol=1.e-3, atol=1.e-5)
        msg = 'There is a large mismatch between calculated semivariance and expected output.' \
              ' Omnidirectional semivariogram.'
        self.assertTrue(are_close, msg)

    def test_calculate_semivariance_zeros_omni(self):
        arr = calculate_semivariance(gen_data.input_zeros,
                                     step_size=gen_data.param_step_size,
                                     max_range=gen_data.param_max_range)
        mean_val = np.mean(arr[:, 1])
        msg = 'Calculated semivariance should be equal to zero if we provide only zeros array.'
        self.assertEqual(mean_val, gen_data.output_zeros, msg)

    def test_calculate_semivariance_omni(self):
        ss = 1
        mr = 2
        smv = calculate_semivariance(armstrong_arr, ss, mr)
        lag1_test_value = smv[0][1]
        expected_output = sem_data.output_armstrong_lag1
        err_msg = f'Calculated semivariance for lag 1 should be equal to {expected_output} for ' \
                  f'the omnidirectional case.'
        self.assertAlmostEqual(lag1_test_value, expected_output, places=1, msg=err_msg)

    # DIRECTIONAL CASES

    def test_calculate_semivariance_WE_lag1(self):
        ss = 1
        mr = 2
        smv = calculate_semivariance(armstrong_arr, ss, mr, direction=0, tolerance=0.1, method='e')
        lag1_test_value = smv[0][1]
        expected_output = sem_data.output_armstrong_ns_lag1
        err_msg = f'Calculated semivariance for lag 1 should be equal to {expected_output} for ' \
                  f'the N-S direction.'
        self.assertAlmostEqual(lag1_test_value, expected_output, places=2, msg=err_msg)

    def test_calculate_semivariance_NS_lag1(self):
        ss = 1
        mr = 2
        smv = calculate_semivariance(armstrong_arr, ss, mr, direction=90, tolerance=0.1, method='e')
        lag1_test_value = smv[0][1]
        expected_output = sem_data.output_armstrong_we_lag1
        err_msg = f'Calculated semivariance for lag 1 should be equal to {expected_output} for ' \
                  f'the W-E direction.'
        self.assertAlmostEqual(lag1_test_value, expected_output, places=2, msg=err_msg)

    def test_calculate_semivariance_NW_SE_lag2(self):
        ss = 1
        mr = 4
        smv = calculate_semivariance(armstrong_arr, ss, mr, direction=135, tolerance=0.01, method='e')
        lag2_test_value = smv[1][1]
        expected_output = sem_data.output_armstrong_nw_se_lag2
        err_msg = f'Calculated semivariance for lag 2 should be equal to {expected_output} for ' \
                  f'the NW-SE direction.'
        self.assertAlmostEqual(lag2_test_value, expected_output, places=2, msg=err_msg)

    def test_calculate_semivariance_NE_SW_lag2(self):
        ss = 1
        mr = 3
        smv = calculate_semivariance(armstrong_arr, ss, mr, direction=45, tolerance=0.01, method='e')
        lag2_test_value = smv[1][1]
        expected_output = sem_data.output_armstrong_ne_sw_lag2
        err_msg = f'Calculated semivariance for lag 2 should be equal to {expected_output} for ' \
                  f'the NE-SW direction.'
        self.assertAlmostEqual(lag2_test_value, expected_output, places=2, msg=err_msg)

    # WEIGHTED CASES

    def test_calculate_weighted_omnidirectional(self):
        _input = gen_data.input_weighted.copy()
        arr = _input[0]
        weight_arr = _input[1]
        ss = 1
        mr = 3
        smv = calculate_semivariance(arr, ss, mr, weights=weight_arr[:, -1])
        expected_output = sem_data.output_weighted
        arr_check = np.allclose(smv, expected_output, rtol=0.1)
        err_msg = f'Given arrays are not equal. Expected output is {expected_output} ' \
                  f'and calculated output is {smv}'
        self.assertTrue(arr_check, err_msg)

    def test_calculate_directional_weighted(self):
        _input = gen_data.input_weighted.copy()
        arr = _input[0]
        weight_arr = _input[1]
        ss = 2
        mr = 7
        smv = calculate_semivariance(arr, ss, mr, weights=weight_arr[:, -1], direction=45, tolerance=0.01, method='e')
        expected_output = sem_data.directional_output_weighted
        arr_check = np.allclose(smv, expected_output, rtol=0.1)
        err_msg = f'Given arrays are not equal. Expected output for directional weighted semivariogram is ' \
                  f'{expected_output} and calculated output is {smv}'
        self.assertTrue(arr_check, err_msg)

    ### RAISES

    def test_raises_value_error(self):
        pts = np.array([[0, 0, np.nan], [1, 1, 2]])

        kwargs = {
            'points': pts,
            'step_size': 1,
            'max_range': 3,
            'direction': 45,
            'tolerance': 0.5
        }

        self.assertRaises(ValueError, calculate_semivariance, **kwargs)


if __name__ == '__main__':
    unittest.main()
