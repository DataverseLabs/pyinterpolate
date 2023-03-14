import unittest
from pyinterpolate.processing.transform.statistics import remove_outliers


class TestRemoveOutliers(unittest.TestCase):

    def test_remove_outliers_z(self):
        # Input - dict, output - dict
        data = {0: [1, 1, 1, 1, 9],
                1: [1, 1, 2, 3, 4, 5],
                2: [0, 6, 4, 6, 6, 4, 5]}
        outliers = [0, 9]

        ou = remove_outliers(data, method='zscore', z_lower_limit=-1.5, z_upper_limit=1.5)

        for k in ou.keys():
            for outlier in outliers:
                self.assertNotIn(outlier, ou[k], 'Outlier detected in dataset. Something went wrong.')

    def test_remove_outliers_iqr(self):
        # Input - list, output - list
        data = [[1, 1, 1, 1, 9],
                [1, 1, 2, 3, 4, 5],
                [0, 6, 4, 6, 6, 4, 5]]
        outliers = [0, 9]

        ou = remove_outliers(data, method='iqr', iqr_lower_limit=1.5, iqr_upper_limit=1.5)

        for values in ou:
            for outlier in outliers:
                self.assertNotIn(outlier, values, 'Outlier detected in dataset. Something went wrong.')
