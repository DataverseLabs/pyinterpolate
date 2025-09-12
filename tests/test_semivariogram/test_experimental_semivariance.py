import numpy as np
import pytest

from pyinterpolate.semivariogram.experimental.experimental_semivariogram import calculate_semivariance


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

    semivariance = calculate_semivariance(
        REFERENCE_INPUT,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE
    )

    expected_output = np.array(
        [
            [1., 4.625, 24.],
            [2., 5.22727273, 22.],
            [3., 6., 20.]
        ]
    )

    assert isinstance(semivariance, np.ndarray)
    assert semivariance.shape == (3, 3)
    assert np.allclose(semivariance, expected_output)


def test_directional_semivariogram():
    try:
        ds = np.load('armstrong_data.npy')
    except FileNotFoundError:
        try:
            ds = np.load('test_semivariogram/armstrong_data.npy')
        except FileNotFoundError:
            ds = np.load('tests/test_semivariogram/armstrong_data.npy')

    STEP_SIZE = 1.5
    MAX_RANGE = 6
    dirvar = calculate_semivariance(
        ds=ds,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        direction=15,
        tolerance=0.25
    )
    assert isinstance(dirvar, np.ndarray)


def test_directional_weighted_semivariogram():
    try:
        ds = np.load('armstrong_data.npy')
    except FileNotFoundError:
        try:
            ds = np.load('test_semivariogram/armstrong_data.npy')
        except FileNotFoundError:
            ds = np.load('tests/test_semivariogram/armstrong_data.npy')

    STEP_SIZE = 1.5
    MAX_RANGE = 6
    REFERENCE_WEIGHTS = np.random.randint(low=1, high=100, size=len(ds))
    dirvar = calculate_semivariance(
        ds=ds,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        direction=15,
        tolerance=0.25,
        custom_weights=REFERENCE_WEIGHTS
    )
    assert isinstance(dirvar, np.ndarray)


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

    semivariance = calculate_semivariance(
        REFERENCE_INPUT,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE
    )

    wsemivariance1 = calculate_semivariance(
        REFERENCE_INPUT,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        custom_weights=REFERENCE_WEIGHTS_ONES
    )

    wsemivariance2 = calculate_semivariance(
        REFERENCE_INPUT,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        custom_weights=REFERENCE_WEIGHTS_RANGE
    )

    assert not np.allclose(semivariance, wsemivariance1)
    assert not np.allclose(wsemivariance1, wsemivariance2)

    with pytest.raises(ValueError):
        # Must raise ValueError if zero-weight is passed
        _ = calculate_semivariance(
            REFERENCE_INPUT,
            step_size=STEP_SIZE,
            max_range=MAX_RANGE,
            custom_weights=REFERENCE_WEIGHTS_ZEROS
        )


def test_weighted_directional_semivariogram():
    try:
        ds = np.load('armstrong_data.npy')
    except FileNotFoundError:
        try:
            ds = np.load('test_semivariogram/armstrong_data.npy')
        except FileNotFoundError:
            ds = np.load('tests/test_semivariogram/armstrong_data.npy')

    STEP_SIZE = 1.5
    MAX_RANGE = 6
    WEIGHTS = np.random.random(size=len(ds))

    semivariance = calculate_semivariance(
        ds,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        direction=15,
        tolerance=0.25,
        custom_weights=WEIGHTS
    )

    assert isinstance(semivariance, np.ndarray)
