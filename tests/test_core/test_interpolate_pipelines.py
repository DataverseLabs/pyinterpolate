import numpy as np
import pandas as pd

from pyinterpolate.core.pipelines.interpolate import (interpolate_points,
                                                      interpolate_points_dask)

try:
    dem = pd.read_csv('sample_data/dem2180.csv')
except FileNotFoundError:
    try:
        dem = pd.read_csv('test_core/sample_data/dem2180.csv')
    except FileNotFoundError:
        dem = pd.read_csv('tests/test_core/sample_data/dem2180.csv')

dem = dem.to_numpy()


def create_model_validation_sets(dataset: np.array, frac=0.1):
    indexes_of_training_set = np.random.choice(range(len(dataset) - 1),
                                               int(frac * len(dataset)),
                                               replace=False)
    training_set = dataset[indexes_of_training_set]
    validation_set = np.delete(dataset, indexes_of_training_set, 0)
    return training_set, validation_set


def test_interpolate_points():
    interp = interpolate_points(
        theoretical_model=
    )
