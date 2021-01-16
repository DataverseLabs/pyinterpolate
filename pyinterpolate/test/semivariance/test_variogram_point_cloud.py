import unittest
import os
import numpy as np
from pyinterpolate.io_ops.read_data import read_point_data
from pyinterpolate.distance.calculate_distances import calc_point_to_point_distance
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import build_variogram_point_cloud


class TestBuildVariogramPointCloud(unittest.TestCase):
    def test_build_variogram_point_cloud(self):
        my_dir = os.path.dirname(__file__)
        path = os.path.join(my_dir, '../sample_data/poland_dem_gorzow_wielkopolski')

        dataset = read_point_data(path, 'txt')

        # Set semivariance params
        distances = calc_point_to_point_distance(dataset[:, :-1])

        maximum_range = np.max(distances)
        number_of_divisions = 10
        step_size = maximum_range / number_of_divisions

        variogram_cloud = build_variogram_point_cloud(dataset, step_size, maximum_range)
        variogram_number_of_points = [705661, 4795616, 7823488, 9114824, 8978790,
                                      7631956, 5248886, 2493818, 653348, 93140]

        for idx, k in enumerate(variogram_cloud.keys()):
            points_number = len(variogram_cloud[k])
            self.assertEqual(points_number,
                             variogram_number_of_points[idx],
                             f'Number of points {points_number} for distance {k} is not'
                             f' equal {variogram_number_of_points[idx]}.')


if __name__ == '__main__':
    unittest.main()
