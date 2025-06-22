import numpy as np
from pyinterpolate.semivariogram.experimental.classes.variogram_cloud import \
    VariogramCloud

from pyinterpolate.semivariogram.experimental.experimental_semivariogram import \
    calculate_semivariance, point_cloud_semivariance

from ._ds import get_armstrong_data


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


def test_zeros():
    zeros_input = REFERENCE_INPUT.copy()
    zeros_input[:, -1] = 0
    cloud_semivariance = point_cloud_semivariance(
        zeros_input,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE
    )
    for lag, semivars in cloud_semivariance.items():
        assert np.sum(semivars) == 0


def test_calculate_semivariance_fn():
    cloud_semivariance = point_cloud_semivariance(
        REFERENCE_INPUT,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE
    )

    expected_keys = [1., 2., 3.]
    expected_array_lengths = [24, 22, 20]

    semivariance = calculate_semivariance(
        REFERENCE_INPUT,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE
    )

    for idx, value in enumerate(expected_keys):
        arr_len = expected_array_lengths[idx]

        assert value in cloud_semivariance.keys()
        assert arr_len == len(cloud_semivariance[value])
        assert np.mean(cloud_semivariance[value] / 2) == semivariance[idx, 1]
        assert len(cloud_semivariance[value]) == semivariance[idx, 2]


def test_variogram_cloud_class():
    vc = VariogramCloud(
        ds=REFERENCE_INPUT,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE
    )
    stats = vc.describe()
    assert stats[1]['count'] == 24
    assert stats[2]['median'] == 9
    assert isinstance(vc, VariogramCloud)


def test_outliers_removal():
    vc1 = VariogramCloud(
        ds=REFERENCE_INPUT,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE
    )
    vc1.remove_outliers(z_lower_limit=-0.5, z_upper_limit=0.5, inplace=True)

    vc2 = VariogramCloud(
        ds=REFERENCE_INPUT,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE
    )

    vc3 = vc2.remove_outliers(z_lower_limit=-0.5, z_upper_limit=0.5)

    print(vc1)
    print(vc2)
    assert vc1.__str__() != vc2.__str__()
    assert isinstance(vc3, VariogramCloud)
    assert vc3.__str__() != vc2.__str__()


# TODO: plotting in a different place, outside the main test suite
# def test_plotting():
#     vc = VariogramCloud(
#         ds=REFERENCE_INPUT,
#         step_size=STEP_SIZE,
#         max_range=MAX_RANGE
#     )
#     assert vc.plot('scatter')
#     assert vc.plot('box')
#     assert vc.plot('violin')


def test_printing():
    vc = VariogramCloud(
        ds=REFERENCE_INPUT,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE
    )
    vc_str = vc.__str__()

    assert len(vc_str) > 0


def test_directional_semivariogram():
    ds = get_armstrong_data()
    _STEP_SIZE = 1.5
    _MAX_RANGE = 6
    dirvar = calculate_semivariance(
        ds=ds,
        step_size=_STEP_SIZE,
        max_range=_MAX_RANGE,
        direction=15,
        tolerance=0.25,
        dir_neighbors_selection_method='e'
    )

    cloud_semivariance = point_cloud_semivariance(
        ds=ds,
        step_size=_STEP_SIZE,
        max_range=_MAX_RANGE,
        direction=15,
        tolerance=0.25,
        dir_neighbors_selection_method='e'
    )

    for idx, _lag in enumerate(list(cloud_semivariance.keys())):
        arr_len = len(cloud_semivariance[_lag])
        assert arr_len == dirvar[idx][-1]
        assert np.mean(cloud_semivariance[_lag] / 2) == dirvar[idx, 1]


# def test_directional_triangular_semivariogram():
#     try:
#         ds = np.load('armstrong_data.npy')
#     except FileNotFoundError:
#         ds = np.load('test_semivariogram/armstrong_data.npy')
#     STEP_SIZE = 1.5
#     MAX_RANGE = 6
#     dirvar = calculate_semivariance(
#         ds=ds,
#         step_size=STEP_SIZE,
#         max_range=MAX_RANGE,
#         direction=15,
#         tolerance=0.25,
#         dir_neighbors_selection_method='t'
#     )
#
#     assert isinstance(dirvar, np.ndarray)
