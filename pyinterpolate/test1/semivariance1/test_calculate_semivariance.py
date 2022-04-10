import unittest
import os
import numpy as np
from numpy.testing import assert_almost_equal
from pyinterpolate.io_ops.read_data import read_point_data
from pyinterpolate.distance.calculate_distances import calc_point_to_point_distance
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import calculate_semivariance
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import calculate_weighted_semivariance
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import calculate_directional_semivariogram


class TestCalculateDirectionalSemivariance(unittest.TestCase):

    def _create_simple_arr(self):
        arr = [8, 6, 4, 3, 6, 5, 7, 2, 8, 9, 5, 6, 3]
        coordinates = [[0, idx] for idx, e in enumerate(arr)]
        data = [[coordinates[idx][1], coordinates[idx][0], v] for idx, v in enumerate(arr)]
        return data

    def _test_case_1(self, data, step, rng, angle):
        semivariance = calculate_directional_semivariogram(data, step, rng, angle)
        equals = semivariance[1:, -1] == 13
        self.assertTrue(equals.all(), 'Number of points is not equal, should be 13 for each lag but 0')
        self.assertCountEqual(semivariance[0], [0, 0, 0], 'Zero lag should have 0 element and 0 semivariance')

    def _test_case_2(self, data, step, rng, angle):
        semivariance = calculate_directional_semivariogram(data, step, rng, angle)
        output = [[4.625, 24],
                  [6, 20],
                  [3.125, 16]]
        t1 = semivariance[2, 1:] == output[0]
        t2 = semivariance[4, 1:] == output[1]
        t3 = semivariance[6, 1:] == output[2]
        t = np.array([t1, t2, t3]).all()
        self.assertTrue(t, 'Semivariance values are not equal expected results')

    def _test_case_3(self, data, step, rng, angle1, angle2):
        semivariance_ew = calculate_directional_semivariogram(data, step, rng, angle1)
        semivariance_we = calculate_directional_semivariogram(data, step, rng, angle2)

        t = (semivariance_ew == semivariance_we).all()
        self.assertTrue(t, 'Semivariance should be the same for direction and its opposite direction')

    def _test_case_4(self, data, step, rng, angle, tolerance):
        semivariance = calculate_directional_semivariogram(data, step, rng, angle, tolerance)
        output = [1.1, 7.31, 100]
        lags = [0, 100, 296, 340, 542, 170, 564, 146, 488, 252]

        t1 = (semivariance[1] == output).all()
        self.assertTrue(t1, 'Semivariance output for the first lag should be equal to: [1.1, 7.31, 100]')

        t2 = (semivariance[:, -1] == lags).all()
        self.assertTrue(t2, 'Number of pairs per lag is not equal expected number of pairs')

    def _test_case_5(self, data, step, rng, angle):
        self.assertRaises(ValueError, calculate_directional_semivariogram,
                          data, step, rng, angle)

    def _test_case_6(self, data, step, rng, angle, tol):
        self.assertRaises(ValueError, calculate_directional_semivariogram,
                          data, step, rng, angle, tol)

    def test_calculate_directional_semivariogram(self):

        # Data prep

        my_dir = os.path.dirname(__file__)
        path = os.path.join(my_dir, '../sample_data/armstrong_data.npy')
        complex_arr = np.load(path)
        simple_arr = self._create_simple_arr()

        step_size = 1.1
        max_range = 10

        ns_dir = 0
        ew_dir = 90
        we_dir = 270
        ne_sw_dir = 30
        tol = 0.5
        tol_wrong = 1.1
        wrong_dir_too_low = -1

        # Test simple arr
        # NS direction tests: semivariance constant across all lags
        self._test_case_1(simple_arr, step_size, max_range, ns_dir)

        # EW direction tests
        self._test_case_2(simple_arr, step_size, max_range, ew_dir)

        # EW == WE test case
        self._test_case_3(simple_arr, step_size, max_range, ew_dir, we_dir)

        # NE-SW test case
        self._test_case_4(complex_arr, step_size, max_range, ne_sw_dir, tol)

        # Wrong dir case
        self._test_case_5(complex_arr, step_size, max_range, wrong_dir_too_low)

        # Wrong tolerance case
        self._test_case_6(complex_arr, step_size, max_range, ne_sw_dir, tol_wrong)


class TestCalculateSemivariance(unittest.TestCase):

    def test_calculate_semivariance(self):
        my_dir = os.path.dirname(__file__)
        path = os.path.join(my_dir, '../sample_data/poland_dem_gorzow_wielkopolski')

        dataset = read_point_data(path, 'txt')

        # Set semivariance params
        distances = calc_point_to_point_distance(dataset[:, :-1])

        maximum_range = np.max(distances)
        number_of_divisions = 10
        step_size = maximum_range / number_of_divisions

        gamma = calculate_semivariance(dataset, step_size, maximum_range)

        output_int = [116, 316, 496, 578, 566, 522, 543, 618, 591]

        self.assertTrue((gamma[:, 1].astype(np.int) == np.array(output_int)).all(),
                        "Integer part of output should be equal to [116, 316, 496, 578, 566, 522, 543, 618, 591]")


    def test_calculate_weighted_semivariance(self):
        my_dir = os.path.dirname(__file__)
        path = os.path.join(my_dir, '../sample_data/poland_dem_gorzow_wielkopolski')

        dataset = read_point_data(path, 'txt')
        new_col = np.arange(1, len(dataset) + 1)

        dataset_weights = np.zeros((dataset.shape[0], dataset.shape[1] + 1))
        dataset_weights[:, :-1] = dataset
        dataset_weights[:, -1] = new_col

        # Set semivariance params
        distances = calc_point_to_point_distance(dataset_weights[:, :-2])

        maximum_range = np.max(distances)
        number_of_divisions = 10
        step_size = maximum_range / number_of_divisions

        gamma_w = calculate_weighted_semivariance(dataset_weights, step_size, maximum_range)

        output_int = [105, 290, 459, 536, 515, 456, 449, 503, 469]

        self.assertTrue((gamma_w[:, 1].astype(np.int) == np.array(output_int)).all(),
                        f"Integer part of output should be equal to {output_int}\n"
                        f"but it is equal to {gamma_w[:, 1].astype(np.int)}")

    ################################################################
    ########## TUTORIAL TESTS AREA - FEEL FREE TO EXPLORE ##########
    ################################################################

    def test_positive_output(self):
        INPUT = np.array([
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

        # Calculate experimental semivariance
        t_step_size = 1.1
        t_max_range = 6

        experimental_semivariance = calculate_semivariance(INPUT, t_step_size, t_max_range)[:, 1]

        boolean_test = (experimental_semivariance >= 0).all()
        self.assertTrue(boolean_test, 'Test failed. Calculated values are below zero which is non-physical.')

    def test_against_expected_value_1(self):

        REFERENCE_INPUT = np.array([
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

        EXPECTED_OUTPUT = np.array([
			[1, 4.625, 24],
			[2, 5.227, 22],
			[3, 6.0, 20],
			[4, 4.444, 18],
			[5, 3.125, 16]
		])

        # Calculate experimental semivariance
        t_step_size = 1
        t_max_range = 6

        experimental_semivariance = calculate_semivariance(REFERENCE_INPUT, t_step_size, t_max_range)

        # Get first five lags
        estimated_output = experimental_semivariance[:6, :]

        # Compare
        err_msg = 'The reference output and the estimated output are too dissimilar, check your algorithm'
        assert_almost_equal(estimated_output, EXPECTED_OUTPUT, 3, err_msg=err_msg)


if __name__ == '__main__':
    unittest.main()
