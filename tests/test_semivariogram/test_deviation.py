import numpy as np
from pyinterpolate import (
    build_experimental_variogram,
    build_theoretical_variogram
)
from pyinterpolate.semivariogram.deconvolution.deviation import Deviation


REFERENCE_INPUT = np.array([
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
STEP_SIZE = 1
MAX_RANGE = 4

EXPERIMENTAL = build_experimental_variogram(
    values=REFERENCE_INPUT[:, -1],
    geometries=REFERENCE_INPUT[:, :-1],
    step_size=STEP_SIZE,
    max_range=MAX_RANGE
)

REGULARIZED_EXPERIMENTAL = build_experimental_variogram(
    values=REFERENCE_INPUT[:, -1] + 2,
    geometries=REFERENCE_INPUT[:, :-1],
    step_size=STEP_SIZE,
    max_range=MAX_RANGE
)

THEORETICAL = build_theoretical_variogram(
    experimental_variogram=EXPERIMENTAL
)


def test_deviation_mrd():
    dv = Deviation(
        theoretical_semivariances=THEORETICAL.lag_yhat_array,
        regularized_semivariances=REGULARIZED_EXPERIMENTAL.lag_semivariance_array,
        method='mrd'
    )
    assert isinstance(dv, Deviation)
    assert len(dv.deviations) == 1
    assert dv.initial_deviation > 0.028
    assert dv.optimal_deviation == dv.deviations[0]


def test_deviation_smrd():
    dv = Deviation(
        theoretical_semivariances=THEORETICAL.lag_yhat_array,
        regularized_semivariances=REGULARIZED_EXPERIMENTAL.lag_semivariance_array,
        method='smrd'
    )
    assert isinstance(dv, Deviation)
    assert len(dv.deviations) == 1
    assert dv.initial_deviation > 0.028
    assert dv.optimal_deviation == dv.deviations[0]


def test_deviation_rmse():
    dv = Deviation(
        theoretical_semivariances=THEORETICAL.lag_yhat_array,
        regularized_semivariances=REGULARIZED_EXPERIMENTAL.lag_semivariance_array,
        method='rmse'
    )
    assert isinstance(dv, Deviation)
    assert len(dv.deviations) == 1
    assert dv.initial_deviation > 0.028
    assert dv.optimal_deviation == dv.deviations[0]


def test_deviation_methods():
    dv_rmse = Deviation(
        theoretical_semivariances=THEORETICAL.lag_yhat_array,
        regularized_semivariances=REGULARIZED_EXPERIMENTAL.lag_semivariance_array,
        method='rmse'
    )
    rmse = dv_rmse.deviations[0]
    dv_smrd = Deviation(
        theoretical_semivariances=THEORETICAL.lag_yhat_array,
        regularized_semivariances=REGULARIZED_EXPERIMENTAL.lag_semivariance_array,
        method='smrd'
    )
    smrd = dv_smrd.deviations[0]
    dv_mrd = Deviation(
        theoretical_semivariances=THEORETICAL.lag_yhat_array,
        regularized_semivariances=REGULARIZED_EXPERIMENTAL.lag_semivariance_array,
        method='mrd'
    )
    mrd = dv_mrd.deviations[0]

    assert rmse != smrd
    assert smrd != mrd
    assert mrd != rmse
