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
        data=DS,
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
