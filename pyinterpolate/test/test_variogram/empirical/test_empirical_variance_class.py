# Core modules
import unittest
# Math modules
import numpy as np
# Package modules
from pyinterpolate.variogram.empirical.covariance import calculate_covariance
from pyinterpolate.variogram.empirical.semivariance import calculate_semivariance
# Tested module
from pyinterpolate.variogram.empirical.experimental_variogram import build_experimental_variogram, \
    ExperimentalVariogram


# Test data
from pyinterpolate.test.test_variogram.empirical.consts import EmpiricalVariogramTestData, EmpiricalVariogramClassData


gen_data = EmpiricalVariogramTestData()
cls_data = EmpiricalVariogramClassData()


class TestEmpiricalSemivariance(unittest.TestCase):

    def test_build_statistics_with_zeros(self):
        variogram_stats = ExperimentalVariogram(
            input_array=gen_data.input_zeros,
            step_size=gen_data.param_step_size,
            max_range=gen_data.param_max_range
        )

        mean_semivar = np.mean(variogram_stats.experimental_semivariances)
        mean_covar = np.mean(variogram_stats.experimental_covariances)
        var_value = variogram_stats.variance

        expected_value = gen_data.output_zeros

        self.assertEqual(mean_semivar, expected_value)
        self.assertEqual(mean_covar, expected_value)
        self.assertEqual(var_value, expected_value)

    def test_output_both(self):

        variogram_stats = ExperimentalVariogram(
            input_array=gen_data.input_data_we,
            step_size=cls_data.param_step_size,
            max_range=cls_data.param_max_range
        )

        # Check __str__() output

        str_output = ''.join(variogram_stats.__str__().split())

        expected_output = ''.join(
            """+-----+--------------------+---------------------+--------------------+
               | lag |    semivariance    |      covariance     |    var_cov_diff    |
               +-----+--------------------+---------------------+--------------------+
               | 1.0 |       4.625        | -0.5434027777777798 | 4.791923487836951  |
               | 2.0 | 5.2272727272727275 | -0.7954545454545454 | 5.0439752555137165 |
               | 3.0 |        6.0         | -1.2599999999999958 | 5.508520710059168  |
               +-----+--------------------+---------------------+--------------------+""".split()
        )

        self.assertEqual(str_output, expected_output, msg='Expected __str__() output is different than '
                                                          'the returned text')

        # Check __repr__() output

        new_variogram_stats = eval(variogram_stats.__repr__())
        new_str_output = ''.join(new_variogram_stats.__str__().split())

        self.assertEqual(new_str_output, expected_output, msg='Expected __str__() output is different than '
                                                              'the returned __repr__() object text')

    def test_single_output_semivariance(self):
        variogram_stats = ExperimentalVariogram(
            input_array=gen_data.input_data_we,
            step_size=cls_data.param_step_size,
            max_range=cls_data.param_max_range,
            is_semivariance=True,
            is_covariance=False,
            is_variance=False
        )
        variogram = calculate_semivariance(
            points=gen_data.input_data_we,
            step_size=cls_data.param_step_size,
            max_range=cls_data.param_max_range
        )

        are_equal = np.array_equal(variogram_stats.experimental_semivariance_array, variogram)
        self.assertTrue(are_equal, msg='Semivariogram calculated with the EmpiricalSemivariogram class and '
                                       'the calculate_semivariance() function must be the same!')

        var_stats_str = ''.join(variogram_stats.__str__().split())
        expected_str = ''.join(
            """ +-----+--------------------+------------+--------------+
                | lag |    semivariance    | covariance | var_cov_diff |
                +-----+--------------------+------------+--------------+
                | 1.0 |       4.625        |    nan     |     nan      |
                | 2.0 | 5.2272727272727275 |    nan     |     nan      |
                | 3.0 |        6.0         |    nan     |     nan      |
                +-----+--------------------+------------+--------------+""".split()
        )

        self.assertEqual(var_stats_str, expected_str, msg='Expected __str__() output is different for semivariance'
                                                          ' than the returned text')

    def test_single_output_covariance(self):
        variogram_stats = ExperimentalVariogram(
            input_array=gen_data.input_data_we,
            step_size=cls_data.param_step_size,
            max_range=cls_data.param_max_range,
            is_semivariance=False,
            is_covariance=True,
            is_variance=True
        )
        covariogram, variance = calculate_covariance(
            gen_data.input_data_we,
            step_size=cls_data.param_step_size,
            max_range=cls_data.param_max_range,
            get_c0=True
        )

        are_equal = np.array_equal(variogram_stats.experimental_covariance_array, covariogram)
        self.assertTrue(are_equal, msg='Covariogram calculated with the EmpiricalSemivariogram class and '
                                       'the calculate_covariance() function must be the same!')

        self.assertEqual(variogram_stats.variance, variance, msg='Variances for the EmpiricalSemivariogram class'
                                                                 ' and the calculate_covariance() function must'
                                                                 ' be the same!')

        var_stats_str = ''.join(variogram_stats.__str__().split())

        expected_str = ''.join(
            """ +-----+--------------+---------------------+--------------------+
                | lag | semivariance |      covariance     |    var_cov_diff    |
                +-----+--------------+---------------------+--------------------+
                | 1.0 |     nan      | -0.5434027777777798 | 4.791923487836951  |
                | 2.0 |     nan      | -0.7954545454545454 | 5.0439752555137165 |
                | 3.0 |     nan      | -1.2599999999999958 | 5.508520710059168  |
                +-----+--------------+---------------------+--------------------+""".split()
        )

        self.assertEqual(var_stats_str, expected_str, msg='Expected __str__() output is different for semivariance'
                                                          ' than the returned text')

    def test_build_variogram_stats(self):
        exp_variogram = build_experimental_variogram(
            input_array=gen_data.input_weighted[0],
            step_size=cls_data.param_step_size,
            max_range=cls_data.param_bounded_max_range,
            weights=gen_data.input_weighted[1][:, -1]
        )

        variogram_stats = ExperimentalVariogram(
            input_array=gen_data.input_weighted[0],
            step_size=cls_data.param_step_size,
            max_range=cls_data.param_bounded_max_range,
            weights=gen_data.input_weighted[1][:, -1]
        )

        are_equal_semivariances = np.array_equal(
            exp_variogram.experimental_semivariances, variogram_stats.experimental_semivariances
        )

        are_equal_covariances = np.array_equal(
            exp_variogram.experimental_covariances, variogram_stats.experimental_covariances
        )

        are_equal_variances = np.equal(exp_variogram.variance, variogram_stats.variance)

        # Assertions

        self.assertTrue(are_equal_variances, msg='Output variance must be the same for EmpiricalSemivariogram class'
                                                 ' and build_experimental_variogram() function.')
        self.assertTrue(are_equal_covariances, msg='Output covariance must be the same for EmpiricalSemivariogram class'
                                                   ' and build_experimental_variogram() function.')
        self.assertTrue(are_equal_semivariances, msg='Output semivariance must be the same for EmpiricalSemivariogram '
                                                     'class and build_experimental_variogram() function.')

    def test_bounded_variogram(self):

        bounded_variogram = build_experimental_variogram(
            input_array=cls_data.input_bounded,
            step_size=cls_data.param_step_size,
            max_range=cls_data.param_bounded_max_range,
        )

        diff = bounded_variogram.variance_covariances_diff
        are_close = np.allclose(bounded_variogram.experimental_semivariances,
                                diff, rtol=2)

        self.assertTrue(are_close, msg='Expeced difference c(0) - c(h) should be close to the '
                                       'semivariances at y(h) for a given dataset.')

    def test_raises_value_error(self):
        pts = np.array([[0, 0, np.nan], [1, 1, 2]])

        kwargs = {
            'input_array': pts,
            'step_size': 1,
            'max_range': 3,
            'direction': 45,
            'tolerance': 0.5
        }

        self.assertRaises(ValueError, ExperimentalVariogram, **kwargs)
