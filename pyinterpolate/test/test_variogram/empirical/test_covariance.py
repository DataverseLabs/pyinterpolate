import os
import unittest
import numpy as np
from pyinterpolate.variogram.empirical.covariance import calculate_covariance

# Imports for tests within this scope
from pyinterpolate.test.test_variogram.empirical.consts import get_armstrong_data, EmpiricalVariogramTestData,\
    EmpiricalCovarianceData


gen_data = EmpiricalVariogramTestData()
cov_data = EmpiricalCovarianceData()
armstrong_arr = get_armstrong_data()

class TestCovariance(unittest.TestCase):

    # OMNIDIRECTIONAL CASES

    def test_variance_c0(self):
        output = calculate_covariance(gen_data.input_data_we,
                                      step_size=gen_data.param_step_size,
                                      max_range=gen_data.param_max_range)
        c0 = output[1]
        expected_output = cov_data.output_variance
        msg = f'Expected variance of the input dataset is {expected_output} but ' \
              f'{c0} was returned.'
        self.assertAlmostEqual(c0, expected_output, 2, msg=msg)

    def test_calculate_covariance_single_row(self):
        output = calculate_covariance(gen_data.input_data_we,
                                      step_size=gen_data.param_step_size,
                                      max_range=gen_data.param_max_range)
        output = output[0]
        expected_output = cov_data.output_we_omni
        are_close = np.allclose(output, expected_output, rtol=1.e-2)
        msg = 'The difference between expected values and calculated covariances are too large, check calculations.'
        self.assertTrue(are_close, msg)

    def test_calculate_covariance_zeros_omni(self):
        output = calculate_covariance(gen_data.input_zeros,
                                      step_size=gen_data.param_step_size,
                                      max_range=gen_data.param_max_range)
        mean_val = np.mean(output[0][:, 1])
        expected_output = gen_data.output_zeros
        msg = 'Calculated covariance should be equal to zero if we provide only zeros array.'
        self.assertEqual(mean_val, expected_output, msg)

    def test_calculate_covariance_omni(self):
        ss = 1
        mr = 2
        c0bool = False
        cov = calculate_covariance(armstrong_arr, ss, mr, get_c0=c0bool)
        lag1_test_value = cov[0][0][1]
        expected_output = cov_data.output_armstrong_omni_lag1
        err_msg = f'Calculated covariance for lag 1 should be equal to {expected_output} for ' \
                  f'the omnidirectional case.'
        self.assertAlmostEqual(lag1_test_value, expected_output, places=2, msg=err_msg)

    # DIRECTIONAL CASES

    def test_calculate_covariance_SN_lag1(self):
        ss = 1
        mr = 2
        c0bool = False
        cov = calculate_covariance(armstrong_arr, ss, mr, direction=0, tolerance=0.01, get_c0=c0bool)
        lag1_test_value = cov[0][0][1]
        expected_output = cov_data.output_armstrong_ns_lag1
        err_msg = f'Calculated covariance for lag 1 should be equal to {expected_output} for ' \
                  f'N-S direction.'
        self.assertAlmostEqual(lag1_test_value, expected_output, places=2, msg=err_msg)

    def test_calculate_semivariance_WE_lag1(self):
        ss = 1
        mr = 2
        c0bool = False
        cov = calculate_covariance(armstrong_arr, ss, mr, direction=90, tolerance=0.1, get_c0=c0bool)
        cov = cov[0]
        lag1_test_value = cov[0][1]
        expected_output = cov_data.output_armstrong_we_lag1
        err_msg = f'Calculated semivariance for lag 1 should be equal to {expected_output} for ' \
                  f'the W-E direction.'
        self.assertAlmostEqual(lag1_test_value, expected_output, places=2, msg=err_msg)

    def test_calculate_semivariance_NW_SE_lag2(self):
        ss = 1
        mr = 4
        c0bool = False
        cov = calculate_covariance(armstrong_arr, ss, mr, direction=135, tolerance=0.01, get_c0=c0bool)
        cov = cov[0]
        lag1_test_value = cov[1][1]
        expected_output = cov_data.output_armstrong_nw_se_lag2
        err_msg = f'Calculated covariance for lag 2 should be equal to {expected_output} for ' \
                  f'the NW-SE direction..'
        self.assertAlmostEqual(lag1_test_value, expected_output, places=2, msg=err_msg)

    def test_calculate_semivariance_NE_SW_lag2(self):
        ss = 1
        mr = 3
        c0bool = False
        cov = calculate_covariance(armstrong_arr, ss, mr, direction=45, tolerance=0.01, get_c0=c0bool)
        cov = cov[0]
        lag1_test_value = cov[1][1]
        expected_output = cov_data.output_armstrong_ne_sw_lag2
        err_msg = f'Calculated covariance for lag 2 should be equal to {expected_output} for ' \
                  f'the NE-SW direction.'
        self.assertAlmostEqual(lag1_test_value, expected_output, places=2, msg=err_msg)

    def test_raises_value_error(self):
        pts = np.array([[0, 0, np.nan], [1, 1, 2]])

        kwargs = {
            'points': pts,
            'step_size': 1,
            'max_range': 3,
            'direction': 45,
            'tolerance': 0.5,
            'get_c0': False
        }

        self.assertRaises(ValueError, calculate_covariance, **kwargs)


if __name__ == '__main__':
    unittest.main()
