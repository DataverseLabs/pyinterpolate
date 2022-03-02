import unittest
import numpy as np
from pyinterpolate.transform.prepare_kriging_data import prepare_kriging_data


class TestPrepareKrigingData(unittest.TestCase):

    def test_prepare_kriging_data(self):
        EXPECTED_NUMBER_OF_NEIGHBORS = 1
        EXPECTED_OUTPUT = np.array([[13, 10, 9, 3]])
        unknown_pos = (10, 10)
        pos = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [13, 10, 9]]
        pos = np.array(pos)
        d = prepare_kriging_data(unknown_pos, pos, neighbors_range=4)
        d = d.astype(np.int)

        test_output = np.array_equal(d, EXPECTED_OUTPUT)
        test_length = len(d) == EXPECTED_NUMBER_OF_NEIGHBORS

        self.assertTrue(test_length, "Length of prepared dataset should be 1 (one neighbour)")
        self.assertTrue(test_output, "Output array should be [[13, 10, 9, 3]]")

    def test_prepare_kriging_data_no_neighb_in_range(self):
        EXPECTED_OUTPUT = np.array([[13, 10, 9, 3]])
        unknown_pos = (10, 10)
        pos = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [13, 10, 9]]
        pos = np.array(pos)
        d = prepare_kriging_data(unknown_pos, pos, neighbors_range=2)
        d = d.astype(np.int)
        test_output = np.array_equal(d, EXPECTED_OUTPUT)
        self.assertTrue(test_output, "Output array should be [[13, 10, 9, 3]]")

if __name__ == '__main__':
    unittest.main()
