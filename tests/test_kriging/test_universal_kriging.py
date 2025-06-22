import unittest
import numpy as np

from pyinterpolate import read_txt, ExperimentalVariogram, TheoreticalVariogram
from pyinterpolate.kriging import UniversalKriging, MultivariateRegression

try:
    dem = read_txt('samples/point_data/txt/pl_dem_epsg2180.txt')
except FileNotFoundError:
    dem = read_txt('../samples/point_data/txt/pl_dem_epsg2180.txt')

def create_model_validation_sets(dataset: np.array, frac=0.1):
    indexes_of_training_set = np.random.choice(range(len(dataset) - 1), int(frac * len(dataset)), replace=False)
    training_set = dataset[indexes_of_training_set]
    validation_set = np.delete(dataset, indexes_of_training_set, 0)
    return training_set, validation_set


class TestUniversalKriging(unittest.TestCase):

    def test_init(self):
        uk = UniversalKriging(observations=dem)
        self.assertIsInstance(uk.observations, np.ndarray)

    def test_get_trend(self):
        uk = UniversalKriging(observations=dem)
        uk.fit_trend()
        self.assertIsInstance(uk.trend_model, MultivariateRegression)
        self.assertIsInstance(uk.trend_values, np.ndarray)

    def test_get_bias(self):
        uk = UniversalKriging(observations=dem)
        uk.fit_trend()
        uk.detrend()
        self.assertIsInstance(uk.bias_values, np.ndarray)

    def test_model_bias(self):
        uk = UniversalKriging(observations=dem)
        uk.fit_trend()
        uk.detrend()
        uk.fit_bias(
            step_size=500, max_range=10000
        )
        self.assertIsInstance(uk.bias_experimental_model, ExperimentalVariogram)
        self.assertIsInstance(uk.bias_model, TheoreticalVariogram)

    def test_predict(self):
        known_values, unknown_points = create_model_validation_sets(dem)
        uk = UniversalKriging(observations=known_values)
        uk.fit_trend()
        uk.detrend()
        uk.fit_bias(
            step_size=500, max_range=10000
        )
        predictions = uk.predict(
            points=unknown_points[:, :-1]
        )
        rmse = np.sqrt(
            np.mean(
                (unknown_points[:, -1] - predictions[:, 0])**2
            )
        )

        # uk.plot_trend_surfaces()

        self.assertIsInstance(
            predictions, np.ndarray
        )
        self.assertGreater(rmse, 0)

