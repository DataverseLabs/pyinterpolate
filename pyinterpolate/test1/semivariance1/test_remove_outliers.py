import unittest
from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import remove_outliers


class TestRemoveOutliers(unittest.TestCase):

    def test_remove_outliers(self):

        data = {0: [1, 1, 1, 1, 9],
                1: [1, 1, 2, 3, 4, 5],
                2: [0, 6, 4, 6, 6, 4, 5]}
        outliers = [0, 9]

        ou = remove_outliers(data, 'both')

        for k in ou.keys():
            for outlier in outliers:
                self.assertNotIn(outlier, ou[k], 'Outlier detected in dataset. Something went wrong.')

    def test_raise_error(self):
        with self.assertRaises(TypeError):
            _ = remove_outliers({0: [0]}, 'tip')


if __name__ == '__main__':
    unittest.main()
