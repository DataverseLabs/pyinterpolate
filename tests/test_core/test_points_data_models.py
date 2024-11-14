import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from pydantic import ValidationError

from pyinterpolate.core.data_models.points import RawPoints, VariogramPoints


REFERENCE_INPUT_LIST = [
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
    ]


REFERENCE_INPUT = np.array(REFERENCE_INPUT_LIST)

GEOPANDAS_INPUT = gpd.GeoDataFrame(data=REFERENCE_INPUT)
GEOPANDAS_INPUT['geometry'] = gpd.points_from_xy(x=REFERENCE_INPUT[:, 0],
                                                 y=REFERENCE_INPUT[:, 1])
GEOPANDAS_INPUT.columns = ['x', 'y', 'val', 'geometry']

PANDAS_INPUT = pd.DataFrame(
    data=REFERENCE_INPUT,
    columns=['xl', 'yl', 'val']
)


def test_raw_points_class():
    rp = RawPoints(
        **{"points": REFERENCE_INPUT}
    )
    assert isinstance(rp.points, np.ndarray)
    assert isinstance(rp, RawPoints)
    assert np.array_equal(rp.points, REFERENCE_INPUT)


def test_kriging_points_class():
    rp = RawPoints(**{"points": REFERENCE_INPUT})
    kp = VariogramPoints(REFERENCE_INPUT)

    assert np.array_equal(kp.points, rp.points)


def test_pass_wrong_number_of_columns():
    with pytest.raises(ValidationError) as _:
        _ = RawPoints(**{"points": REFERENCE_INPUT[:, :2]})


def test_list_input():
    kp = VariogramPoints(REFERENCE_INPUT_LIST)
    assert isinstance(kp.points, np.ndarray)


def test_geopandas_input():
    kp = VariogramPoints(GEOPANDAS_INPUT[['geometry', 'val']])
    assert isinstance(kp.points, np.ndarray)


def test_pandas_input():
    kp = VariogramPoints(PANDAS_INPUT)
    assert isinstance(kp.points, np.ndarray)
