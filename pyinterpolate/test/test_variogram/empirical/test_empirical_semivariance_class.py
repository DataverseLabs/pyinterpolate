# Core modules
import unittest
# Math modules
import numpy as np
# Package modules
from pyinterpolate.variogram.empirical.covariance import calculate_covariance
from pyinterpolate.variogram.empirical.semivariance import calculate_semivariance
# Tested module
from pyinterpolate.variogram.empirical.empirical_semivariogram import build_experimental_variogram, \
    EmpiricalSemivariogram

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
REFERENCE_INPUT_BOUNDED = np.array([
    [0, 0, 2],
    [1, 0, 4],
    [2, 0, 6],
    [3, 0, 8],
    [4, 0, 10],
    [5, 0, 12],
    [6, 0, 14],
    [7, 0, 12],
    [8, 0, 10],
    [9, 0, 8],
    [10, 0, 6],
    [11, 0, 4],
    [12, 0, 2]
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

# CONSTS
STEP_SIZE = 1
MAX_RANGE = 4
BOUNDED_RANGE = (len(REFERENCE_INPUT_BOUNDED) / 2) - 1


class TestEmpiricalSemivariance(unittest.TestCase):

    def test_build_statistics_with_zeros(self):
        variogram_stats = EmpiricalSemivariogram(
            input_array=REFERENCE_INPUT_ZEROS,
            step_size=STEP_SIZE,
            max_range=MAX_RANGE
        )

        mean_semivar = np.mean(variogram_stats.experimental_semivariance[:, 1])
        mean_covar = np.mean(variogram_stats.experimental_covariance[:, 1])
        var_value = variogram_stats.variance

        expected_value = 0

        self.assertEqual(mean_semivar, expected_value)
        self.assertEqual(mean_covar, expected_value)
        self.assertEqual(var_value, expected_value)

    def test_output_both(self):

        variogram_stats = EmpiricalSemivariogram(
            input_array=REFERENCE_INPUT_WE,
            step_size=STEP_SIZE,
            max_range=MAX_RANGE
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
        variogram_stats = EmpiricalSemivariogram(
            input_array=REFERENCE_INPUT_WE,
            step_size=STEP_SIZE,
            max_range=MAX_RANGE,
            is_semivariance=True,
            is_covariance=False,
            is_variance=False
        )
        variogram = calculate_semivariance(
            points=REFERENCE_INPUT_WE,
            step_size=STEP_SIZE,
            max_range=MAX_RANGE
        )

        are_equal = np.array_equal(variogram_stats.experimental_semivariance, variogram)
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
        variogram_stats = EmpiricalSemivariogram(
            input_array=REFERENCE_INPUT_WE,
            step_size=STEP_SIZE,
            max_range=MAX_RANGE,
            is_semivariance=False,
            is_covariance=True,
            is_variance=True
        )
        covariogram, variance = calculate_covariance(
            points=REFERENCE_INPUT_WE,
            step_size=STEP_SIZE,
            max_range=MAX_RANGE,
            get_c0=True
        )

        are_equal = np.array_equal(variogram_stats.experimental_covariance, covariogram)
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
            input_array=REFERENCE_INPUT_WEIGHTED[0],
            step_size=STEP_SIZE,
            max_range=MAX_RANGE,
            weights=REFERENCE_INPUT_WEIGHTED[1][:, -1]
        )

        variogram_stats = EmpiricalSemivariogram(
            input_array=REFERENCE_INPUT_WEIGHTED[0],
            step_size=STEP_SIZE,
            max_range=MAX_RANGE,
            weights=REFERENCE_INPUT_WEIGHTED[1][:, -1]
        )

        are_equal_semivariances = np.array_equal(
            exp_variogram.experimental_semivariance, variogram_stats.experimental_semivariance
        )

        are_equal_covariances = np.array_equal(
            exp_variogram.experimental_covariance, variogram_stats.experimental_covariance
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
            input_array=REFERENCE_INPUT_BOUNDED,
            step_size=STEP_SIZE,
            max_range=BOUNDED_RANGE
        )

        diff = bounded_variogram.variance - bounded_variogram.experimental_covariance[:, 1]
        are_close = np.allclose(bounded_variogram.experimental_semivariance[:, 1],
                                diff, rtol=2)

        self.assertTrue(are_close, msg='Expeced difference c(0) - c(h) should be close to the '
                                       'semivariances at y(h) for a given dataset.')
