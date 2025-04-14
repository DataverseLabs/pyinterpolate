import numpy as np
import pandas as pd
import pytest

from pyinterpolate.semivariogram.experimental.classes.experimental_variogram import ExperimentalVariogram
from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.centroid_poisson_kriging import CentroidPoissonKrigingInput
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from tests.test_semivariogram.sample_data.dataprep import CANCER_DATA_WITH_CENTROIDS, POINT_SUPPORT_DATA


BLOCKS = Blocks(**CANCER_DATA_WITH_CENTROIDS)

PS = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=BLOCKS,
        points_value_column=POINT_SUPPORT_DATA['value_column_name'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column_name']
    )

EXPERIMENTAL = ExperimentalVariogram(
        ds=BLOCKS.representative_points_array(),
        step_size=40000,
        max_range=300001
    )

THEO = TheoreticalVariogram()
THEO.autofit(
    experimental_variogram=EXPERIMENTAL,
    sill=150
)


EXPERIMENTAL_DIR = ExperimentalVariogram(
    ds=BLOCKS.representative_points_array(),
    step_size=40000,
    max_range=300001,
    direction=15,
    tolerance=0.2
)
THEO_DIR = TheoreticalVariogram()
THEO_DIR.autofit(
    experimental_variogram=EXPERIMENTAL_DIR,
    sill=150,
    models_group='linear'
)


THEO_PARAMS = {"experimental_variogram": None,
               "nugget": 0.0,
               "sill": 176.3272376248095,
               "rang": 180000,
               "variogram_model_type": "spherical",
               "direction": None,
               "spatial_dependence": None,
               "spatial_index": None,
               "yhat": None,
               "errors": None}
THEO_FROM_REG = TheoreticalVariogram()
THEO_FROM_REG.from_dict(THEO_PARAMS)

DS_COLUMNS = ['points', 'values', 'distances', 'angular_differences', 'block_id']


def test_centroid_poisson_kriging_input_class():
    indexes = BLOCKS.block_indexes

    cpki = CentroidPoissonKrigingInput(
        block_id=indexes[-5],
        point_support=PS,
        semivariogram_model=THEO_FROM_REG
    )

    assert isinstance(cpki, CentroidPoissonKrigingInput)
    assert set(cpki.ds.columns) == set(DS_COLUMNS)


    pk_input = cpki.pk_input
    assert isinstance(pk_input, pd.DataFrame)

    # raise ValueError
    with pytest.raises(ValueError) as _:
        coordinates = cpki.coordinates

    with pytest.raises(ValueError) as _:
        distances = cpki.distances

    with pytest.raises(ValueError) as _:
        kriging_input = cpki.kriging_input

    with pytest.raises(ValueError) as _:
        neighbors_indexes = cpki.neighbors_indexes

    with pytest.raises(ValueError) as _:
        values = cpki.values

    cpki.select_neighbors(
        max_range=200000,
        min_number_of_neighbors=4,
        select_all_possible_neighbors=False
    )

    coordinates = cpki.coordinates
    distances = cpki.distances
    kriging_input = cpki.kriging_input
    neighbors_indexes = cpki.neighbors_indexes
    values = cpki.values

    print('')
    print(cpki.kriging_input.head())

    assert isinstance(coordinates, np.ndarray)
    assert isinstance(distances, np.ndarray)
    assert isinstance(kriging_input, pd.DataFrame)
    assert isinstance(neighbors_indexes, np.ndarray)
    assert isinstance(values, np.ndarray)
    assert len(kriging_input) == len(neighbors_indexes) == len(values) == len(coordinates) == len(distances)
    assert len(kriging_input) <= 4  # select_all_possible_neighbors=False
    assert len(kriging_input) != len(pk_input)


def test_centroid_poisson_kriging_input_class_directional_variant():
    indexes = BLOCKS.block_indexes

    cpki = CentroidPoissonKrigingInput(
        block_id=indexes[-5],
        point_support=PS,
        semivariogram_model=THEO_DIR
    )

    assert isinstance(cpki, CentroidPoissonKrigingInput)
    assert set(cpki.ds.columns) == set(DS_COLUMNS)


    pk_input = cpki.pk_input
    assert isinstance(pk_input, pd.DataFrame)

    # raise ValueError
    with pytest.raises(ValueError) as _:
        coordinates = cpki.coordinates

    with pytest.raises(ValueError) as _:
        distances = cpki.distances

    with pytest.raises(ValueError) as _:
        kriging_input = cpki.kriging_input

    with pytest.raises(ValueError) as _:
        neighbors_indexes = cpki.neighbors_indexes

    with pytest.raises(ValueError) as _:
        values = cpki.values

    cpki.select_neighbors(
        max_range=200000,
        min_number_of_neighbors=4,
        select_all_possible_neighbors=False
    )

    coordinates = cpki.coordinates
    distances = cpki.distances
    kriging_input = cpki.kriging_input
    neighbors_indexes = cpki.neighbors_indexes
    values = cpki.values
    angles = cpki.angles

    assert isinstance(coordinates, np.ndarray)
    assert isinstance(distances, np.ndarray)
    assert isinstance(kriging_input, pd.DataFrame)
    assert isinstance(neighbors_indexes, np.ndarray)
    assert isinstance(values, np.ndarray)
    assert len(kriging_input) == len(neighbors_indexes) == len(values) == len(coordinates) == len(distances) == len(angles)
    assert len(kriging_input) <= 4  # select_all_possible_neighbors=False
    assert len(kriging_input) != len(pk_input)
    for angle in angles:
        assert angle >= -360
        assert angle <= 360
