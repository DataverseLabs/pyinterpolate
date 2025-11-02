import numpy as np
import pytest

from pyinterpolate.semivariogram.experimental.classes.experimental_variogram import build_experimental_variogram


def test_omnidirectional_semivariogram():
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

    semivariance = build_experimental_variogram(
        ds=REFERENCE_INPUT,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE
    )
    semivariance = semivariance.semivariances

    expected_output = np.array(
        [
            [4.625, 5.22727273, 6.]
        ]
    )

    assert isinstance(semivariance, np.ndarray)
    assert semivariance.shape == (3,)
    assert np.allclose(semivariance,
                       expected_output)


def test_weighted_omnidirectional_semivariogram():
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

    REFERENCE_WEIGHTS_ONES = np.ones(13)
    REFERENCE_WEIGHTS_RANGE = np.linspace(1, 2, len(REFERENCE_WEIGHTS_ONES))
    REFERENCE_WEIGHTS_ZEROS = np.zeros(13)

    STEP_SIZE = 1
    MAX_RANGE = 4

    # scenario 1 - outputs different with weighted and non-weighted

    semivariance = build_experimental_variogram(
        ds=REFERENCE_INPUT,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE
    )
    semivariance = semivariance.semivariances

    wsemivariance1 = build_experimental_variogram(
        ds=REFERENCE_INPUT,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        custom_weights=REFERENCE_WEIGHTS_ONES
    )
    wsemivariance1 = wsemivariance1.semivariances

    wsemivariance2 = build_experimental_variogram(
        ds=REFERENCE_INPUT,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        custom_weights=REFERENCE_WEIGHTS_RANGE
    )
    wsemivariance2 = wsemivariance2.semivariances

    assert not np.allclose(semivariance, wsemivariance1)
    assert not np.allclose(wsemivariance1, wsemivariance2)

    with pytest.raises(ValueError):
        # Must raise ValueError if zero-weight is passed
        _ = build_experimental_variogram(
            ds=REFERENCE_INPUT,
            step_size=STEP_SIZE,
            max_range=MAX_RANGE,
            custom_weights=REFERENCE_WEIGHTS_ZEROS
        )


def test_omnidirectional_semivariogram_sep_geom():
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

    semivariance = build_experimental_variogram(
        values=REFERENCE_INPUT[:, -1],
        geometries=REFERENCE_INPUT[:, :-1],
        step_size=STEP_SIZE,
        max_range=MAX_RANGE
    )
    semivariance = semivariance.semivariances

    expected_output = np.array(
        [
            [4.625, 5.22727273, 6.]
        ]
    )

    assert isinstance(semivariance, np.ndarray)
    assert semivariance.shape == (3,)
    assert np.allclose(semivariance,
                       expected_output)


def test_weighted_omnidirectional_semivariogram_sep_geom():
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

    REFERENCE_WEIGHTS_ONES = np.ones(13)
    REFERENCE_WEIGHTS_RANGE = np.linspace(1, 2, len(REFERENCE_WEIGHTS_ONES))
    REFERENCE_WEIGHTS_ZEROS = np.zeros(13)

    STEP_SIZE = 1
    MAX_RANGE = 4

    # scenario 1 - outputs different with weighted and non-weighted

    semivariance = build_experimental_variogram(
        values=REFERENCE_INPUT[:, -1],
        geometries=REFERENCE_INPUT[:, :-1],
        step_size=STEP_SIZE,
        max_range=MAX_RANGE
    )
    semivariance = semivariance.semivariances

    wsemivariance1 = build_experimental_variogram(
        values=REFERENCE_INPUT[:, -1],
        geometries=REFERENCE_INPUT[:, :-1],
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        custom_weights=REFERENCE_WEIGHTS_ONES
    )
    wsemivariance1 = wsemivariance1.semivariances

    wsemivariance2 = build_experimental_variogram(
        values=REFERENCE_INPUT[:, -1],
        geometries=REFERENCE_INPUT[:, :-1],
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        custom_weights=REFERENCE_WEIGHTS_RANGE
    )
    wsemivariance2 = wsemivariance2.semivariances

    assert not np.allclose(semivariance, wsemivariance1)
    assert not np.allclose(wsemivariance1, wsemivariance2)

    with pytest.raises(ValueError):
        # Must raise ValueError if zero-weight is passed
        _ = build_experimental_variogram(
            values=REFERENCE_INPUT[:, -1],
            geometries=REFERENCE_INPUT[:, :-1],
            step_size=STEP_SIZE,
            max_range=MAX_RANGE,
            custom_weights=REFERENCE_WEIGHTS_ZEROS
        )
