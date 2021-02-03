import unittest
import os
import numpy as np
from pyinterpolate.io_ops import read_point_data


class TestReadData(unittest.TestCase):

	def test_read_data(self):
		my_dir = os.path.dirname(__file__)
		path_to_the_data = os.path.join(my_dir, '../sample_data/poland_dem_gorzow_wielkopolski')
		data = read_point_data(path_to_the_data, 'txt')

		# Check if data type is ndarray
		check_ndarray = isinstance(data, np.ndarray)
		self.assertTrue(check_ndarray, "Instance of a data type should be numpy array")

		# Check dimensions
		self.assertEqual(data.shape[1], 3, "Shape of data should be 3 - x, y, value")


if __name__ == '__main__':
	unittest.main()
