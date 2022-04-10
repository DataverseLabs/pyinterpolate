import unittest
from pyinterpolate.transform.set_areal_weights import set_areal_weights
import numpy as np


class TestArealWeightsSet(unittest.TestCase):

    def test_set_areal_weighs(self):

        rows_no = 4
        data_array_1 = np.random.random(size=(rows_no, 5))
        ids_range = np.arange(0, rows_no)
        data_array_1[:, 0] = ids_range

        da2 = []
        sums = []

        for pt in ids_range:
            x = np.random.random(size=(rows_no * 8, 3))
            row = [pt, x]
            summed = np.sum(x[:, -1])
            da2.append(row)
            sums.append(summed)

        data_array_2 = np.array(da2)

        new_set = set_areal_weights(data_array_1, data_array_2)

        for idx, s in enumerate(sums):

            # Check sums
            self.assertEqual(int(s), int(new_set[idx, -1]))

            # Check compatibility with the base ds
            self.assertTrue((data_array_1[idx, 2:] == new_set[idx, :-1]).all())


if __name__ == '__main__':
    unittest.main()
