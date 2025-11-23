import numpy as np
from pyinterpolate.semivariogram.experimental.classes.directional_variogram import DirectionalVariogram


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

    directional_variogram = DirectionalVariogram(
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        ds=ds
    )
    variograms = directional_variogram.get()

    assert isinstance(variograms, dict)
    assert {"ISO", "NS", "WE", "NE-SW", "NW-SE"} == set(variograms.keys())


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
    directional_variogram = DirectionalVariogram(
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        ds=ds,
        custom_weights=REFERENCE_WEIGHTS
    )
    variograms = directional_variogram.get()

    assert isinstance(variograms, dict)
    assert {"ISO", "NS", "WE", "NE-SW", "NW-SE"} == set(variograms.keys())


def test_directional_semivariogram_sep_geoms():
    try:
        ds = np.load('armstrong_data.npy')
    except FileNotFoundError:
        try:
            ds = np.load('test_semivariogram/armstrong_data.npy')
        except FileNotFoundError:
            ds = np.load('tests/test_semivariogram/armstrong_data.npy')

    STEP_SIZE = 1.5
    MAX_RANGE = 6

    directional_variogram = DirectionalVariogram(
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        values=ds[:, -1],
        geometries=ds[:, :-1]
    )
    variograms = directional_variogram.get()

    assert isinstance(variograms, dict)
    assert {"ISO", "NS", "WE", "NE-SW", "NW-SE"} == set(variograms.keys())


def test_directional_weighted_semivariogram_sep_geoms():
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
    directional_variogram = DirectionalVariogram(
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        values=ds[:, -1],
        geometries=ds[:, :-1],
        custom_weights=REFERENCE_WEIGHTS
    )
    variograms = directional_variogram.get()

    assert isinstance(variograms, dict)
    assert {"ISO", "NS", "WE", "NE-SW", "NW-SE"} == set(variograms.keys())
