import unittest
import numpy as np
from pyinterpolate.variogram.empirical.cloud import build_variogram_point_cloud

from pyinterpolate.test.test_variogram.empirical.consts import get_armstrong_data, EmpiricalVariogramTestData,\
    EmpiricalSemivarianceData


gen_data = EmpiricalVariogramTestData()
sem_data = EmpiricalSemivarianceData()
armstrong_arr = get_armstrong_data()


class TestVariogramPointCloud(unittest.TestCase):

    # OMNIDIRECTIONAL CASES
    def test_instance(self):
        cloud = build_variogram_point_cloud(
            gen_data.input_data_we,
            step_size=gen_data.param_step_size,
            max_range=gen_data.param_max_range
        )
        self.assertIsInstance(cloud, dict)

    def test_output_we_omni(self):
        cloud = build_variogram_point_cloud(
            gen_data.input_data_we,
            step_size=gen_data.param_step_size,
            max_range=gen_data.param_max_range
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
        expected_output = sem_data.output_we_omni
        are_close = np.allclose(arr, expected_output, rtol=1.e-3, atol=1.e-5)
        msg = 'There is a large mismatch between calculated semivariance and expected output.' \
              ' Omnidirectional semivariogram.'
        self.assertTrue(are_close, msg)

    def test_zeros(self):
        cloud = build_variogram_point_cloud(
            gen_data.input_zeros,
            step_size=gen_data.param_step_size,
            max_range=gen_data.param_max_range
        )
        msg = 'For array with zeros you should always obtain variance between points equal to zero!'
        expected_output = gen_data.output_zeros
        for lag, smv_arr in cloud.items():
            mval = np.mean(smv_arr)
            self.assertEqual(mval, expected_output, msg=msg)

    # DIRECTIONAL CASES
    def test_directional_we_lag1(self):
        smvs = build_variogram_point_cloud(input_array=armstrong_arr,
                                           step_size=1,
                                           max_range=2,
                                           direction=90,
                                           tolerance=0.01)
        lag1_test_values = smvs[1]
        smv = np.mean(lag1_test_values) / 2

        expected_output = sem_data.output_armstrong_we_lag1
        err_msg = f'Calculated semivariance for lag 1 should be equal to {expected_output} for ' \
                  f'the N-S direction.'
        self.assertAlmostEqual(smv, expected_output, places=2, msg=err_msg)

    def test_calculate_semivariance_NE_SW_lag2(self):
        smvs = build_variogram_point_cloud(input_array=armstrong_arr,
                                           step_size=1,
                                           max_range=3,
                                           direction=45,
                                           tolerance=0.01)

        lagx_test_values = np.nan  # It will throw error when no assigned
        for lag, values in smvs.items():
            if lag == 2:
                lagx_test_values = values.copy()
                break

        smv = np.mean(lagx_test_values) / 2
        expected_output = sem_data.output_armstrong_ne_sw_lag2
        err_msg = f'Calculated semivariance for lag 1 should be equal to {expected_output} for ' \
                  f'the NE-SW direction.'
        self.assertAlmostEqual(smv, expected_output, places=2, msg=err_msg)

    def test_raises_value_error(self):
        pts = np.array([[0, 0, np.nan], [1, 1, 2]])

        kwargs = {
            'input_array': pts,
            'step_size': 1,
            'max_range': 3,
            'direction': 45,
            'tolerance': 0.5
        }

        self.assertRaises(ValueError, build_variogram_point_cloud, **kwargs)


if __name__ == '__main__':
    unittest.main()
