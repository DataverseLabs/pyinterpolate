import numpy as np

from pyinterpolate.semivariogram.indicator.indicator import (
    code_indicators,
    select_variogram_thresholds
)

DEM = np.random.random(size=(1000, 3))
DEM_VALUES = DEM[:, -1]
DEM_GEOMETRIES = DEM[:, :-1]
THRESHOLDS = select_variogram_thresholds(
    ds=DEM_VALUES,
    n_thresh=5
)

STEP_R = 0.1
MX_RNG = 0.6


def test_code_indicators():

    indicators = code_indicators(
        thresholds=THRESHOLDS,
        ds=DEM
    )
    assert isinstance(indicators, np.ndarray)
    for row in indicators:
        assert row[0] < row[-1]


def test_code_indicators_sep_geom():

    indicators = code_indicators(
        thresholds=THRESHOLDS,
        values=DEM_VALUES,
        geometries=DEM_GEOMETRIES
    )
    assert isinstance(indicators, np.ndarray)
    for row in indicators:
        assert row[0] < row[-1]
