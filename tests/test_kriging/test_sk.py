import numpy as np

from pyinterpolate.kriging.point.simple import simple_kriging
from .point_kriging_ds.dataprep import build_armstrong_ds, build_random_ds, build_zeros_ds


def test_armstrong_omnidirectional():
    dataset = build_armstrong_ds()
    ds = dataset['ds']
    variogram = dataset['theo_omni']

    unknown_point = [ds[:, 0].std(), ds[:, 1].std()]
    pmean = np.mean(ds[:, -1])

    kriged = simple_kriging(
        theoretical_model=variogram,
        known_locations=ds,
        unknown_locations=unknown_point,
        process_mean=pmean
    )
    assert len(kriged) == 1
    assert len(kriged[0]) == 4


def test_armstrong_directional():
    dataset = build_armstrong_ds()
    ds = dataset['ds']
    variogram = dataset['theo_dir']

    unknown_point = [ds[:, 0].std(), ds[:, 1].std()]
    pmean = np.mean(ds[:, -1])

    kriged = simple_kriging(
        theoretical_model=variogram,
        known_locations=ds,
        unknown_locations=unknown_point,
        process_mean=pmean
    )
    assert len(kriged) == 1
    assert len(kriged[0]) == 4


def test_random_omnidirectional():
    dataset = build_random_ds()
    ds = dataset['ds']
    variogram = dataset['theo_omni']

    unknown_point = [ds[:, 0].std(), ds[:, 1].std()]
    pmean = np.mean(ds[:, -1])

    kriged = simple_kriging(
        theoretical_model=variogram,
        known_locations=ds,
        unknown_locations=unknown_point,
        process_mean=pmean
    )
    assert len(kriged) == 1
    assert len(kriged[0]) == 4


def test_random_directional():
    dataset = build_random_ds()
    ds = dataset['ds']
    variogram = dataset['theo_dir']

    unknown_point = [ds[:, 0].std(), ds[:, 1].std()]
    pmean = np.mean(ds[:, -1])

    kriged = simple_kriging(
        theoretical_model=variogram,
        known_locations=ds,
        unknown_locations=unknown_point,
        process_mean=pmean
    )
    assert len(kriged) == 1
    assert len(kriged[0]) == 4


def test_zeros_omnidirectional():
    dataset = build_zeros_ds()
    ds = dataset['ds']
    variogram = dataset['theo_omni']

    unknown_point = [1, 1]
    pmean = 10

    kriged = simple_kriging(
        theoretical_model=variogram,
        known_locations=ds,
        unknown_locations=unknown_point,
        process_mean=pmean
    )
    assert len(kriged) == 1
    assert len(kriged[0]) == 4
