import unittest
import numpy as np

from pyinterpolate.kriging.point_kriging.kriging import Krige

from pyinterpolate.semivariance.semivariogram_estimation.calculate_semivariance import calculate_semivariance
from pyinterpolate.semivariance.semivariogram_fit.fit_semivariance import TheoreticalSemivariogram
from pyinterpolate.distance.calculate_distances import calc_point_to_point_distance


class SetKrigingModel:

    def __init__(self, data, smv_model):
        self.dataset = data
        self.semivariogram_model = smv_model

        self.kriging_model = Krige(self.semivariogram_model, self.dataset)


class SetSemivariogramModel:

    def __init__(self, positions, number_of_steps):
        self.data = positions
        self.steps = number_of_steps

        distances = calc_point_to_point_distance(self.data[:, :-1])
        maximum_range = np.max(distances)
        step_size = maximum_range / self.steps

        self.semivariance = calculate_semivariance(self.data, step_size, maximum_range)
        self.theoretical_semivariance = TheoreticalSemivariogram(self.data, self.semivariance)
        self.theoretical_semivariance.find_optimal_model(number_of_ranges=self.steps)


class TestKriging(unittest.TestCase):

    @staticmethod
    def _test_ordinary_kriging(kriging_model, unknown_loc):
        km = kriging_model.ordinary_kriging(unknown_loc, test_anomalies=False)
        return int(km[0])

    @staticmethod
    def _test_simple_kriging(kriging_model, unknown_loc, mean):
        km = kriging_model.simple_kriging(unknown_loc, global_mean=mean, test_anomalies=False)
        return int(km[0])

    def test_kriging(self):
        # Set datasets
        unknown_pos = (10, 10)
        pos = [[11, 1, 1], [23, 2, 2], [33, 3, 3], [14, 44, 4], [13, 10, 9], [12, 55, 35], [11, 9, 7]]
        pos = np.array(pos)

        sk_mean = np.mean(pos[:, -1])

        # Set semivariogram and kriging model
        s_model = SetSemivariogramModel(pos, 5)
        k_model = SetKrigingModel(pos, s_model.theoretical_semivariance)

        kok = self._test_ordinary_kriging(k_model.kriging_model, unknown_pos)
        ksk = self._test_simple_kriging(k_model.kriging_model, unknown_pos, sk_mean)

        EXPECTED_VALUE_ORDINARY = 7
        EXPECTED_VALUE_SIMPLE = 6

        self.assertEqual(kok, EXPECTED_VALUE_ORDINARY, f"Ordinary Kriging value should be {EXPECTED_VALUE_ORDINARY}")
        self.assertEqual(ksk, EXPECTED_VALUE_SIMPLE, f"Simple Kriging value should be {EXPECTED_VALUE_SIMPLE}")


if __name__ == '__main__':
    unittest.main()
