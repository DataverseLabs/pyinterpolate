import unittest
import numpy as np
from pyinterpolate.data_processing.data_preparation.select_values_in_range import select_values_in_range


class TestValuesInRange(unittest.TestCase):

	def test_select_values_in_range(self):
	    dataset = [[1, 2, 3],
	               [4, 5, 6],
	               [1, 9, 9]]
	    output = (np.array([1, 1, 1]), np.array([0, 1, 2]))
	    x = select_values_in_range(dataset, 5, 2)

	    self.assertTrue((x[0] == output[0]).all(), "Output row indices should be [1, 1, 1]")
	    self.assertTrue((x[1] == output[1]).all(), "Output col indices should be [0, 1, 2]")


if __name__ == '__main__':
    unittest.main()
