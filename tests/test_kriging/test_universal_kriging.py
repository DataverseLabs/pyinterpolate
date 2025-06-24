import numpy as np
import pandas as pd

from pyinterpolate.kriging.point.universal import UniversalKriging, \
    MultivariateRegression
from pyinterpolate import ExperimentalVariogram, TheoreticalVariogram

try:
    dem = pd.read_csv('point_kriging_ds/dem2180.csv')
except FileNotFoundError:
    dem = pd.read_csv('test_kriging/point_kriging_ds/dem2180.csv')
dem = dem.to_numpy()


def create_model_validation_sets(dataset: np.array, frac=0.1):
    indexes_of_training_set = np.random.choice(range(len(dataset) - 1),
                                               int(frac * len(dataset)),
                                               replace=False)
    training_set = dataset[indexes_of_training_set]
    validation_set = np.delete(dataset, indexes_of_training_set, 0)
    return training_set, validation_set


def test_init():
    uk = UniversalKriging(
        known_points=dem
    )
    assert isinstance(uk.known_points, np.ndarray)


def test_get_trend():
    uk = UniversalKriging(
        known_points=dem
    )
    uk.fit_trend()
    assert isinstance(uk.trend_model, MultivariateRegression)
    assert isinstance(uk.trend_values, np.ndarray)


def test_get_bias():
    uk = UniversalKriging(
        known_points=dem
    )
    uk.fit_trend()
    uk.detrend()
    assert isinstance(uk.bias_values, np.ndarray)


def test_model_bias():
    uk = UniversalKriging(
        known_points=dem
    )
    uk.fit_trend()
    uk.detrend()
    uk.fit_bias(
        step_size=500, max_range=10000
    )
    assert isinstance(uk.bias_experimental_model, ExperimentalVariogram)
    assert isinstance(uk.bias_model, TheoreticalVariogram)


def test_predict():
    known_values, unknown_points = create_model_validation_sets(dem)
    uk = UniversalKriging(
        known_points=dem
    )
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
            (unknown_points[:, -1] - predictions[:, 0]) ** 2
        )
    )

    # uk.plot_trend_surfaces()

    assert isinstance(
        predictions, np.ndarray
    )
    assert rmse > 0
