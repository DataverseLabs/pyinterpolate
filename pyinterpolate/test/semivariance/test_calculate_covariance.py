import unittest
import os
import numpy as np
from pyinterpolate.io.read_data import read_point_data
from pyinterpolate.distance.calculate_distances import calc_point_to_point_distance
from pyinterpolate.semivariance.semivariogram_estimation.calculate_covariance import calculate_covariance


class TestCalculateCovariance(unittest.TestCase):

    def test_calculate_covariance(self):
        my_dir = os.path.dirname(__file__)
        path = os.path.join(my_dir, '../sample_data/poland_dem_gorzow_wielkopolski')

        dataset = read_point_data(path, 'txt')

        # Set semivariance params
        distances = calc_point_to_point_distance(dataset[:, :-1])

        maximum_range = np.max(distances)
        number_of_divisions = 10
        step_size = maximum_range / number_of_divisions
        lags = np.arange(0, maximum_range, step_size)

        gamma = calculate_covariance(dataset, lags, step_size)

        output_int = [368, 197, 19,  0,  0,  0,  0,  0,  0,  0]

        check_equality = (gamma[:, 1].astype(np.int) == np.array(output_int)).all()
        self.assertTrue(check_equality, "Int if output array is not equal to [368, 197, 19,  0,  0,  0,  0,  0,  0,  0]")


if __name__ == '__main__':
    unittest.main()
