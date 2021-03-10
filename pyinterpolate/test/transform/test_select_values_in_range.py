import os
import unittest
import numpy as np
from pyinterpolate.transform.select_values_in_range import select_values_in_range, check_points_within_ellipse


class TestCheckPointsWithinEllipse(unittest.TestCase):

    def _test_case_1(self, pt, arr, rng, last_rng, rad, tol):
        ellipse1 = check_points_within_ellipse(pt, arr, rng[0], last_rng[0], rad, tol)
        ellipse2 = check_points_within_ellipse(pt, arr, rng[1], last_rng[1], rad, tol)
        t = (ellipse1 * ellipse2).any()
        self.assertFalse(t, "Ellipses with common center but different ranges shouldn't have the same points")

    def _test_case_2(self, pt, arr, rng, last_rng, rad, tol):
        ell = check_points_within_ellipse(pt, arr, rng, last_rng, rad, tol)
        pts = arr[ell]
        output = [[2, 5, 25],
                  [3, 5, 20],
                  [5, 3, 16],
                  [6, 3, 13]]
        output = np.array(output)
        t = (pts == output).all()
        self.assertTrue(t, 'Wrong points are chosen, they are not lying inside ellipse')

    def test_check_points_within_ellipse(self):
        my_dir = os.path.dirname(__file__)
        path = os.path.join(my_dir, '../sample_data/armstrong_data.npy')
        complex_arr = np.load(path)

        origin_point = complex_arr[36]

        # Test case 1: two ellipses with the same origin but different range
        # Ellipse 2 must contain other points than Ellipse 1
        rngs = [1.5, 2.5]
        last_rgs = [0, 1.5]
        radians = 0
        tolerance = 0

        self._test_case_1(origin_point, complex_arr, rngs, last_rgs, radians, tolerance)

        # Test case 2: Specific points within ellipse
        rng = 2
        lrng = 0
        radians = np.pi * 0.33
        tolerance = 0.2

        self._test_case_2(origin_point, complex_arr, rng, lrng, radians, tolerance)


class TestValuesInRange(unittest.TestCase):

    def test_select_values_in_range(self):
        dataset = [[1, 2, 3],
                   [4, 5, 6],
                   [1, 9, 9]]
        output = (np.array([1, 1]), np.array([1, 2]))
        x = select_values_in_range(dataset, 5, 2)

        self.assertTrue((x[0] == output[0]).all(), "Output row indices should be [1, 1]")
        self.assertTrue((x[1] == output[1]).all(), "Output col indices should be [1, 2]")


if __name__ == '__main__':
    unittest.main()
