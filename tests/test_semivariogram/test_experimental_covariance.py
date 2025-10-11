from dataclasses import dataclass
import numpy as np

from pyinterpolate.semivariogram.experimental.experimental_covariogram import calculate_covariance


@dataclass
class EmpiricalCovarianceData:
    output_we_omni = np.array([
        [1, -0.543, 24],
        [2, -0.795, 22],
        [3, -1.26, 20]
    ])

    output_armstrong_we_lag1 = 4.643
    output_armstrong_ns_lag1 = 9.589
    output_armstrong_nw_se_lag2 = 4.551
    output_armstrong_ne_sw_lag2 = 6.331
    output_armstrong_omni_lag1 = 6.649


def test_omnidirectional_covariogram():
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

    covariance = calculate_covariance(
        REFERENCE_INPUT,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE
    )

    expected_output = EmpiricalCovarianceData.output_we_omni

    assert isinstance(covariance, np.ndarray)
    assert covariance.shape == (3, 3)
    assert np.allclose(covariance, expected_output, rtol=3)


def test_directional_covariogram():
    try:
        ds = np.load('armstrong_data.npy')
    except FileNotFoundError:
        try:
            ds = np.load('test_semivariogram/armstrong_data.npy')
        except FileNotFoundError:
            ds = np.load('tests/test_semivariogram/armstrong_data.npy')

    STEP_SIZE = 1.5
    MAX_RANGE = 6
    dirvar = calculate_covariance(
        ds=ds,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        direction=15,
        tolerance=0.25
    )
    assert isinstance(dirvar, np.ndarray)
