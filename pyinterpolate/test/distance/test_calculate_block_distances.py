import unittest
import numpy as np
from pyinterpolate.distance.calculate_distances import calc_block_to_block_distance


class TestCalcBlock2BlockDistance(unittest.TestCase):

    def test_calc_block_to_block_distance(self):
        # Test block_to_block_distances

        areal_test_data = [
            ['id0', [[0, 1, 1],
                     [0, 2, 1],
                     [0, 3, 1]]
             ],
            ['id1', [[1, 0, 2],
                     [1, 1, 2]]],

            ['id2', [[2, 1, 3],
                     [2, 2, 4],
                     [2, 3, 4]]]
        ]

        x = calc_block_to_block_distance(areal_test_data)
        self.assertEqual(int(np.sum(x[0])), 12, "Block to block distance is not calculated correctly, should be 12")


if __name__ == '__main__':
    unittest.main()
