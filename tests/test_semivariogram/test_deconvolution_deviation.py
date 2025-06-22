# Mean Relative Difference: ``|Theoretical - Regularized| / Regularized``
import numpy as np

from pyinterpolate.semivariogram.deconvolution.deviation import mean_relative_difference, \
    symmetric_mean_relative_difference


def test_mean_relative_difference_deviation():
    regularized = np.array([1, 2, 3, 4, 5, 6])
    modeled = np.array([2, 3, 4, 5, 6, 7])
    deviation = mean_relative_difference(y_exp=regularized,
                                         y_init=modeled)
    assert isinstance(deviation, float)
    assert deviation > 0


def test_zero_denominator_mean_relative_difference_deviation():
    regularized = np.array([0, 2, 3, 3])
    modeled = np.array([1, 1, 2, 3])
    deviation = mean_relative_difference(
        y_exp=regularized,
        y_init=modeled
    )

    assert isinstance(deviation, float)
    assert deviation == np.inf


def test_symm_mean_relative_difference_deviation():
    regularized = np.array([1, 2, 3, 4, 5, 6])
    modeled = np.array([2, 3, 4, 5, 6, 7])
    deviation = symmetric_mean_relative_difference(y_exp=regularized,
                                                   y_init=modeled)
    assert isinstance(deviation, float)
    assert deviation > 0


def test_zero_denominator_symm_mean_relative_difference_deviation():
    regularized = np.array([0, 2, 3, 3])
    modeled = np.array([1, 1, 2, 3])
    deviation = symmetric_mean_relative_difference(
        y_exp=regularized,
        y_init=modeled
    )

    assert isinstance(deviation, float)
    assert deviation != np.inf
    assert deviation >= 0
