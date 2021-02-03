import unittest
import numpy as np

from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import calc_semivariance_from_pt_cloud


class TestCalcSemivarianceFromPtCloud(unittest.TestCase):
    def test_calc_semivariance_from_pt_cloud(self):
        data = {0: [1, 1, 1, 1, 1],
                1: [0, 1, 2, 3, 4, 5],
                2: [3, 3, 4, 6, 3, 4, 5]}

        output_vec = np.array([[0, 1, 5], [1, 2.5, 6], [2, 4, 7]])

        s = calc_semivariance_from_pt_cloud(data)

        self.assertTrue((s == output_vec).all(), 'Calculated semivariance is not equal to expected array')


if __name__ == '__main__':
    unittest.main()
