import os
import unittest
import geopandas as gpd
import numpy as np
from pyinterpolate.variogram.empirical.semivariance import calculate_semivariance

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
			[0, 0, 13],
			[1, 4.625, 24],
			[2, 5.227, 22],
			[3, 6.0, 20],
			[4, 4.444, 18],
			[5, 3.125, 16]
		])


class TestSemivariance(unittest.TestCase):

    def test_calculate_semivariance(self):
        my_dir = os.path.dirname(__file__)
        path_to_the_data = os.path.join(my_dir, '../../samples/point_data/shapefile/test_points_pyinterpolate.shp')
        df = gpd.read_file(path_to_the_data)
        print(df.head())


if __name__ == '__main__':
    unittest.main()