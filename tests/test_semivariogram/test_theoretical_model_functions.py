import numpy as np
from typing import Callable, Iterable

from pyinterpolate.semivariogram.theoretical.variogram_models.models import (
    circular_model,
    cubic_model,
    exponential_model,
    gaussian_model,
    linear_model,
    power_model,
    spherical_model
)

from ._ds import TheoreticalVariogramModelsTestData

VARIANCES = TheoreticalVariogramModelsTestData()
LAGS = VARIANCES.lags


def build_model(model_fn: Callable, lags: Iterable, nugget: float, sill: float, rang: float):
    mdl = model_fn(
        lags=lags,
        nugget=nugget,
        sill=sill,
        rang=rang
    )
    return mdl


def variance_0_mean_test(_model: Callable):
    expected_variance_mean = 0
    cm_000010 = build_model(_model, LAGS, 0, 0, 0)
    modeled_variance_mean = np.mean(cm_000010)
    return modeled_variance_mean == expected_variance_mean


def variance_1_mean_test(_model: Callable):
    expected_variance_mean = 1
    cm_100010 = build_model(_model, LAGS, 1, 0, 0)
    modeled_variance_mean = np.mean(cm_100010)
    return modeled_variance_mean == expected_variance_mean


def variance_expected_val_mean_test(_model: Callable, val: float):
    cm_111010 = build_model(_model, LAGS, 1, 1, 1)
    modeled_variance_mean = np.mean(cm_111010)
    return modeled_variance_mean == val


def variance_close_to_val_mean_test(_model: Callable, val: float):
    cm_1110010 = build_model(_model, LAGS, 1, 1, 10)
    modeled_variance_mean = float(np.mean(cm_1110010))
    return np.allclose(
        modeled_variance_mean, val, rtol=3
    )


def nugget_test(_model: Callable):
    rrange = VARIANCES.rang_random
    rnugget = VARIANCES.nugget_random
    rsill = VARIANCES.sill_random
    cm_random = build_model(_model,
                            LAGS,
                            rnugget,
                            rsill,
                            rrange)
    return cm_random[0] == rnugget


def multiple_tests(_model: Callable,
                   expected_val: float,
                   close_to_val: float,
                   expected_as_close=False) -> bool:
    assert variance_0_mean_test(_model)

    # n=1, s=0, r=0, l=(0:10)
    assert variance_1_mean_test(_model)

    # n=1, s=1, r=1, l=(0:10)
    if expected_as_close:
        assert variance_close_to_val_mean_test(_model, expected_val)
    else:
        assert variance_expected_val_mean_test(_model, expected_val)

    # n=1, s=1, r=10, l=(0:10)
    assert variance_close_to_val_mean_test(_model, close_to_val)

    # n=random, s=random, r=random, l=(0:10)
    assert nugget_test(_model)

    return True


def test_circular():
    expected_t1 = 1.9
    expected_t2_close_to = 1.525
    assert multiple_tests(circular_model, expected_t1, expected_t2_close_to)


def test_cubic():
    expected_t1 = 1.9
    expected_t2_close_to = 1.585
    assert multiple_tests(cubic_model, expected_t1, expected_t2_close_to)


def test_exponential():
    expected_t1 = 1.84
    expected_t2_close_to = 1.336
    assert multiple_tests(exponential_model, expected_t1, expected_t2_close_to, expected_as_close=True)


def test_gaussian():
    expected_t1 = 1.86
    expected_t2_close_to = 1.222
    assert multiple_tests(gaussian_model, expected_t1, expected_t2_close_to, expected_as_close=True)


def test_linear():
    expected_t1 = 1.9
    expected_t2_close_to = 1.45
    assert multiple_tests(linear_model, expected_t1, expected_t2_close_to)


def test_power():
    expected_t1 = 1.9
    expected_t2_close_to = 1.285
    assert multiple_tests(power_model, expected_t1, expected_t2_close_to)


def test_spherical():
    expected_t1 = 1.9
    expected_t2_close_to = 1.57375
    assert multiple_tests(spherical_model, expected_t1, expected_t2_close_to)
