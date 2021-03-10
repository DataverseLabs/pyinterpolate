import unittest
import os
import numpy as np
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
        self.assertTrue(equals.all())
        self.assertCountEqual(semivariance[0], [0, 0, 0])


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
        nw_se_dir = 200
        wrong_dir_too_much = 400
        wrong_dir_too_low = -1

        # Test simple arr
        # NS direction tests: semivariance constant across all lags
        self._test_case_1(simple_arr, step_size, max_range, ns_dir)

        # EW 




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

        output_int = [51, 207, 416, 552, 579, 544, 517, 589, 619, 540]

        self.assertTrue((gamma[:, 1].astype(np.int) == np.array(output_int)).all(),
                        "Integer part of output should be equal to [51, 207, 416, 552, 579, 544, 517, 589, 619, 540]")


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

        output_int = [46, 189, 384, 512, 534, 486, 438, 488, 493, 414]

        self.assertTrue((gamma_w[:, 1].astype(np.int) == np.array(output_int)).all(),
                        "Integer part of output should be equal to [46, 189, 384, 512, 534, 486, 438, 488, 493, 414]")


if __name__ == '__main__':
    unittest.main()
