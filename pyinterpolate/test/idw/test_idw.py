import unittest
import numpy as np

from pyinterpolate.idw.idw import inverse_distance_weighting


class TestIDW(unittest.TestCase):

    def test_idw(self):
        unknown_pos = (10, 10)
        pos1 = [[11, 1, 1], [23, 2, 2], [33, 3, 3], [14, 44, 4], [13, 10, 9], [12, 55, 35], [11, 9, 7]]
        pos2 = [[11, 1, 1], [23, 2, 2], [33, 3, 3], [14, 44, 4], [10, 10, 999], [12, 55, 35], [11, 9, 7]]

        u_val1 = inverse_distance_weighting(np.array(pos1),
                                            np.array(unknown_pos),
                                            -1, 0.5)

        u_val2 = inverse_distance_weighting(np.array(pos1),
                                           np.array(unknown_pos),
                                           3)

        u_val3 = inverse_distance_weighting(np.array(pos2),
                                            np.array(unknown_pos),
                                            3)


        # Test case 1: u_val1 > u_val2 > 7

        self.assertGreater(u_val1, u_val2, 'Value of power 0.5 should be greater than value of power 3.')
        self.assertGreater(u_val2, 7, f'Value {u_val2} should be greater than 7.')

        # Test case 2: 999

        self.assertEqual(u_val3, 999, f'Zero distance test failed.')