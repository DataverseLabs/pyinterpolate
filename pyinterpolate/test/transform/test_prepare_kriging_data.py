import unittest
import numpy as np
from pyinterpolate.transform.prepare_kriging_data import prepare_kriging_data


class TestPrepareKrigingData(unittest.TestCase):

    def test_prepare_kriging_data(self):
        dataset_length = 3
        unknown_pos = (10, 10)
        pos = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [13, 10, 9]]
        pos = np.array(pos)
        d = prepare_kriging_data(unknown_pos, pos, dataset_length)
        vals = np.array([[13, 10, 9, 3], [4, 4, 4, 8], [3, 3, 3, 9]])
        d = d.astype(np.int)

        test_output = (d == vals).all()

        test_length = len(d) == dataset_length

        self.assertTrue(test_length, "Length of prepared dataset should be 3")
        self.assertTrue(test_output, "Output array should be [[13, 10, 9, 3], [4, 4, 4, 8], [3, 3, 3, 9]]")


if __name__ == '__main__':
    unittest.main()
