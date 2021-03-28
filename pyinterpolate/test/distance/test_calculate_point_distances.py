import unittest
import numpy as np
from pyinterpolate.distance.calculate_distances import calc_point_to_point_distance


class TestCalcPoint2PointDistance(unittest.TestCase):

  def test_calc_point_to_point_distance(self):
      coords = [(10, -10),
                (11, -8.9711),
                (12.2, -13),
                (11, -10.0)]

      d = calc_point_to_point_distance(coords)

      test_arr = np.array([[0, 1, 3, 1],
                           [1, 0, 4, 1],
                           [3, 4, 0, 3],
                           [1, 1, 3, 0]])

      sum_d = np.sum(d.astype(int))
      sum_test_arr = np.sum(test_arr)
      self.assertEqual(sum_d, sum_test_arr, "Distances between points are not calculated correctly")


if __name__ == '__main__':
    unittest.main()
