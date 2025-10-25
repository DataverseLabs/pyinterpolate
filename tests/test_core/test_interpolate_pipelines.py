import geopandas as gpd
import numpy as np
import pandas as pd

from pyinterpolate import build_experimental_variogram, \
    build_theoretical_variogram
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

    train, test = create_model_validation_sets(dem)
    geometries = gpd.points_from_xy(x=train[:, 0],
                                    y=train[:, 1])

    step_size = 500  # meters
    max_range = 10000  # meters

    exp_var = build_experimental_variogram(
        ds=train,
        step_size=step_size,
        max_range=max_range
    )
    theo_var = build_theoretical_variogram(
        experimental_variogram=exp_var
    )

    interp = interpolate_points(
        theoretical_model=theo_var,
        unknown_locations=test[:, :2],
        known_values=train[:, -1],
        known_geometries=geometries
    )
    assert isinstance(interp, np.ndarray)


def test_interpolate_points_dask():

    train, test = create_model_validation_sets(dem)
    geometries = gpd.points_from_xy(x=train[:, 0],
                                    y=train[:, 1])

    step_size = 500  # meters
    max_range = 10000  # meters

    exp_var = build_experimental_variogram(
        ds=train,
        step_size=step_size,
        max_range=max_range
    )
    theo_var = build_theoretical_variogram(
        experimental_variogram=exp_var
    )

    interp = interpolate_points_dask(
        theoretical_model=theo_var,
        unknown_locations=test[:, :2],
        known_values=train[:, -1],
        known_geometries=geometries,
        number_of_workers=4,
        allow_approximate_solutions=True
    )
    assert isinstance(interp, np.ndarray)
