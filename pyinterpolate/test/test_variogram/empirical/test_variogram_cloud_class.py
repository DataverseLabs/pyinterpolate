# Core modules
import unittest
# Math modules
import numpy as np
# Package modules
from pyinterpolate.variogram.empirical.semivariance import calculate_semivariance
# Tested module
from pyinterpolate.variogram.empirical.variogram_cloud import VariogramCloud
# Test data
from pyinterpolate.test.test_variogram.empirical.consts import get_armstrong_data, EmpiricalVariogramTestData,\
    EmpiricalSemivarianceData, VariogramPointCloudClassData


gen_data = EmpiricalVariogramTestData()
smv_data = EmpiricalSemivarianceData()
cls_data = VariogramPointCloudClassData()
armstrong_arr = get_armstrong_data()


class TestEmpiricalSemivariance(unittest.TestCase):

    def test_build_statistics_with_zeros(self):
        variogram_cloud = VariogramCloud(
            input_array=gen_data.input_zeros,
            step_size=gen_data.param_step_size,
            max_range=gen_data.param_max_range
        )

        expected_value = gen_data.output_zeros

        for lag in variogram_cloud.lags:
            vals = variogram_cloud.experimental_point_cloud[lag]
            self.assertEqual(np.sum(vals), expected_value)

    def test_output_values_against_semivariance(self):

        variogram_cloud = VariogramCloud(
            input_array=gen_data.input_data_we,
            step_size=gen_data.param_step_size,
            max_range=gen_data.param_max_range
        )

        semivariances = calculate_semivariance(points=gen_data.input_data_we,
                                               step_size=gen_data.param_step_size,
                                               max_range=gen_data.param_max_range)

        for lag in variogram_cloud.lags:
            smvs = semivariances[semivariances[:, 0] == lag][0]
            sval = smvs[1]
            spairs = smvs[2]

            cloud_vals = variogram_cloud.experimental_point_cloud[lag]
            cloud_pairs = len(cloud_vals)
            cloud_smv = np.mean(cloud_vals) / 2  # semivariance from pt cloud
            msg_vals = 'We expect equal semivariances given by calculate_semivariance() function and VariogramCloud' \
                       ' class.'
            msg_pts = 'We expect equal point pairs number given by calculate_semivariance() function and ' \
                      'VariogramCloud class.'
            self.assertAlmostEqual(sval, cloud_smv, places=3, msg=msg_vals)
            self.assertEqual(spairs, cloud_pairs, msg=msg_pts)

    def test_class_print_output(self):
        variogram_cloud = VariogramCloud(
            input_array=gen_data.input_data_we,
            step_size=gen_data.param_step_size,
            max_range=gen_data.param_max_range
        )

        str_output = ''.join(variogram_cloud.__str__()).split()
        self.assertEqual(str_output, cls_data.class_output)

    def test_description(self):
        variogram_cloud = VariogramCloud(
            input_array=armstrong_arr,
            step_size=gen_data.param_step_size,
            max_range=gen_data.param_max_range
        )

        desc = variogram_cloud.describe()

        self.assertEqual(desc[1], cls_data.describe_output[1])

    def test__repr__(self):
        variogram_cloud = VariogramCloud(
            input_array=gen_data.input_data_we,
            step_size=gen_data.param_step_size,
            max_range=gen_data.param_max_range
        )

        new_cloud = eval(variogram_cloud.__repr__())

        msg = 'Variogram cloud generated with __repr__() method should be the same as the initial variogram cloud'
        self.assertEqual(variogram_cloud.describe(),
                         new_cloud.describe(),
                         msg=msg)

    # TODO: test plots (future)
