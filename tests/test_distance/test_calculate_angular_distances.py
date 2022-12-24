import unittest
import numpy as np

from pyinterpolate.distance.distance import calculate_angular_distance


ANGLES = np.array([0, 180, 30, 90, -120, -290, -30, 360, -360])

class TestCalculateAngularDistance(unittest.TestCase):

    def test_zero(self):
        my_angle = 0
        angles = calculate_angular_distance(ANGLES, my_angle)
        expected_angles = [0, 180, 30, 90, 120, 70, 30, 0, 0]
        for idx, angle in enumerate(expected_angles):
            self.assertAlmostEqual(angle, angles[idx], places=6)

    def test_360(self):
        my_angle = 360
        angles = calculate_angular_distance(ANGLES, my_angle)
        expected_angles = [0, 180, 30, 90, 120, 70, 30, 0, 0]
        for idx, angle in enumerate(expected_angles):
            self.assertAlmostEqual(angle, angles[idx], places=6)

    def test_plus_30(self):
        my_angle = 30
        angles = calculate_angular_distance(ANGLES, my_angle)
        expected_angles = [30, 150, 0, 60, 150, 40, 60, 30, 30]
        for idx, angle in enumerate(expected_angles):
            self.assertAlmostEqual(angle, angles[idx], places=6)

    def test_minus_30(self):
        my_angle = -30
        angles = calculate_angular_distance(ANGLES, my_angle)
        expected_angles = [30, 150, 60, 120, 90, 100, 0, 30, 30]
        for idx, angle in enumerate(expected_angles):
            self.assertAlmostEqual(angle, angles[idx], places=6)

    def test_plus_240(self):
        my_angle = 240
        angles = calculate_angular_distance(ANGLES, my_angle)
        expected_angles = [120, 60, 150, 150, 0, 170, 90, 120, 120]
        for idx, angle in enumerate(expected_angles):
            self.assertAlmostEqual(angle, angles[idx], places=6)

    def test_minus_240(self):
        my_angle = -240
        angles = calculate_angular_distance(ANGLES, my_angle)
        expected_angles = [120, 60, 90, 30, 120, 50, 150, 120, 120]
        for idx, angle in enumerate(expected_angles):
            self.assertAlmostEqual(angle, angles[idx], places=6)
