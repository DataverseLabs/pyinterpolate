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
