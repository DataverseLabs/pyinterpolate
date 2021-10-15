import unittest
import os
import numpy as np
from pyinterpolate.io.read_data import read_csv, read_txt


class TestReadData(unittest.TestCase):

	def test_read_txt(self):
		my_dir = os.path.dirname(__file__)
		path_to_the_data = os.path.join(my_dir, '../sample_data/point_data/txt/pl_dem.txt')
		data = read_txt(path_to_the_data, 'txt')

		# Check if data type is ndarray
		check_ndarray = isinstance(data, np.ndarray)
		self.assertTrue(check_ndarray, "Instance of a data type should be numpy array")

		# Check dimensions
		self.assertEqual(data.shape[1], 3, "Shape of data should be 3 - x, y, value")


if __name__ == '__main__':
	unittest.main()
