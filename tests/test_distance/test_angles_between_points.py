import unittest

import numpy as np

from pyinterpolate.distance.distance import calc_angles


class TestCalcPoint2PointsAngle(unittest.TestCase):

    def test_1(self):
        vec_a = (1, 0)
        vec_b = (1, -1)
        angles = calc_angles(vec_b, vec_a)
        expected_angles = [45]
        for idx, ang in enumerate(angles):
            self.assertEqual(ang, expected_angles[idx])

    def test_2(self):
        vec_a = (1, 1)
        vec_b = np.array(
            [[0, 1], [2, 2], [0., 0.], [-1, -1]]
        )
        angles = calc_angles(vec_b, vec_a)
        expected_angles = [315, 0, 45, 180]
        for idx, ang in enumerate(angles):
            self.assertEqual(ang, expected_angles[idx])

    def test_3(self):
        vec_b = np.array(
            [[0, 1], [2, 2], [0., 0.], [-1, -1]]
        )
        angles = calc_angles(vec_b)
        expected_angles = [90, 45, 0, 225]
        for idx, ang in enumerate(angles):
            self.assertEqual(ang, expected_angles[idx])

    def test_4(self):
        import pandas as pd

        sample = 'samples/point_data/csv/meuse_grid.csv'
        df = pd.read_csv(sample, usecols=['x', 'y'])
        vec_b = df.values
        angles = calc_angles(vec_b)

        # This is wrong, other point should be chosen as an origin
        all_equal = np.allclose(angles, 61.5, rtol=0.1)
        self.assertTrue(all_equal)

        # Now we choose origin in the middle of a set
        origin = np.mean(vec_b, axis=0)
        new_angles = calc_angles(vec_b, origin=origin)

        self.assertAlmostEqual(np.mean(new_angles), 164, places=1)
        self.assertAlmostEqual(np.std(new_angles), 96.1, places=1)
