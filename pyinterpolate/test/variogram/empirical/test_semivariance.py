import os
import unittest
import geopandas as gpd
import numpy as np
from pyinterpolate.variogram.empirical.semivariance import calculate_semivariance

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

# EXPECTED OUTPUTS
EXPECTED_OUTPUT_WE_OMNI = np.array([
    [0, 0, 13],
	[1, 4.625, 24],
	[2, 5.227, 22],
	[3, 6.0, 20],
	[4, 4.444, 18],
	[5, 3.125, 16]
])
EXPECTED_OUTPUT_ZEROS = 0

# CONSTS
STEP_SIZE = 1
MAX_RANGE = 6


class TestSemivariance(unittest.TestCase):

    def test_calculate_semivariance_we_omni(self):
        output = calculate_semivariance(REFERENCE_INPUT_WE, step_size=STEP_SIZE, max_range=MAX_RANGE)
        are_close = np.allclose(output, EXPECTED_OUTPUT_WE_OMNI, rtol=1.e-3, atol=1.e-5)
        msg = 'There is a large mismatch between calculated semivariance and expected output.' \
              ' Omnidirectional semivariogram.'
        self.assertTrue(are_close, msg)

    def test_calculate_semivariance_zeros_omni(self):
        output = calculate_semivariance(REFERENCE_INPUT_ZEROS, step_size=STEP_SIZE, max_range=MAX_RANGE)
        mean_val = np.mean(output[:, 1])
        msg = 'Calculated semivariance should be equal to zero if we provide only zeros array.'
        self.assertEqual(mean_val, EXPECTED_OUTPUT_ZEROS, msg)

    def test_calculate_semivariance_omni(self):
        my_dir = os.path.dirname(__file__)
        path_to_the_data = os.path.join(my_dir, '../../samples/point_data/shapefile/test_points_pyinterpolate.shp')
        df = gpd.read_file(path_to_the_data)
        arr = df[['geometry', 'value']].values
        output = calculate_semivariance(arr, step_size=)
        print(arr)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()