import numpy as np

from pyinterpolate.viz.raster import set_dimensions, interpolate_raster
from tests.test_kriging.point_kriging_ds.dataprep import build_armstrong_ds


DATASET = build_armstrong_ds()
DS = DATASET['ds']
VARIOGRAM = DATASET['theo_omni']


def test_set_dimensions():
    dims = set_dimensions(
        DS[:, 0],
        DS[:, 1],
        10
    )
    assert isinstance(dims[0], np.ndarray)
    assert isinstance(dims[1], np.ndarray)
    assert isinstance(dims[2], list)


def test_interpolate_raster():
    interpolated = interpolate_raster(
        known_locations=DS,
        dim=50,
        number_of_neighbors=4,
        semivariogram_model=VARIOGRAM,
        allow_approx_solutions=False,
    )

    assert isinstance(interpolated, dict)
    assert 'result' in interpolated
    assert 'error' in interpolated
    assert 'params' in interpolated
    assert isinstance(interpolated['result'], np.ndarray)
    assert isinstance(interpolated['error'], np.ndarray)
    assert isinstance(interpolated['params'], dict)
    assert interpolated['result'].shape == (51, 51)


def test_interpolate_raster_sep_geom():
    interpolated = interpolate_raster(
        known_values=DS[:, -1],
        known_geometries=DS[:, :-1],
        dim=50,
        number_of_neighbors=4,
        semivariogram_model=VARIOGRAM,
        allow_approx_solutions=False,
    )

    assert isinstance(interpolated, dict)
    assert 'result' in interpolated
    assert 'error' in interpolated
    assert 'params' in interpolated
    assert isinstance(interpolated['result'], np.ndarray)
    assert isinstance(interpolated['error'], np.ndarray)
    assert isinstance(interpolated['params'], dict)
    assert interpolated['result'].shape == (51, 51)


def test_simple_examples_dataset():
    input_data = np.array([
        [0, 0, 8],
        [1, 0, 6],
        [2, 0, 4],
        [3, 0, 3],
        [4, 0, 6],
        [5, 0, 5],
        [6, 0, 7],
        [7, 0, 2],
        [8, 0, 8],
        [9, 0, 9],
        [10, 0, 5],
        [11, 0, 6],
        [12, 0, 3]
    ])
    results = interpolate_raster(
        known_values=input_data[:, -1],
        known_geometries=input_data[:, :-1],
        dim=20
    )

    # import json
    # print(json.dumps(results, indent=2, default=str))
    assert isinstance(results, dict)
    assert 'result' in results
    assert 'params' in results
    assert 'error' in results
