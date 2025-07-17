import os

import numpy as np

from pyinterpolate.semivariogram.experimental.classes.experimental_variogram import ExperimentalVariogram
from pyinterpolate.semivariogram.theoretical.theoretical import build_theoretical_variogram
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram

from ._ds import get_armstrong_data, TheoreticalVariogramTestData

VARIOGRAM_DATA = TheoreticalVariogramTestData()
ARMSTRONG_DATA = get_armstrong_data()

ZEROS_VARIOGRAM = ExperimentalVariogram(ds=VARIOGRAM_DATA.input_zeros,
                                        step_size=VARIOGRAM_DATA.param_step_size,
                                        max_range=VARIOGRAM_DATA.param_max_range)

WE_VARIOGRAM = ExperimentalVariogram(VARIOGRAM_DATA.input_data_we,
                                     step_size=VARIOGRAM_DATA.param_step_size,
                                     max_range=VARIOGRAM_DATA.param_max_range)

ARMSTRONG_VARIOGRAM = ExperimentalVariogram(ARMSTRONG_DATA,
                                            step_size=1,
                                            max_range=6)

ARMSTRONG_VARIOGRAM_DIRECTIONAL = ExperimentalVariogram(ARMSTRONG_DATA,
                                                        step_size=1.2,
                                                        max_range=6,
                                                        direction=135,
                                                        tolerance=0.02)


def test_zero_case():
    _sill = 10
    _range = 10
    _theo = build_theoretical_variogram(
        ZEROS_VARIOGRAM, 'linear', sill=_sill, rang=_range
    )

    expected_rmse = 3.32

    assert np.allclose(expected_rmse, _theo.rmse, rtol=2)


def test_zero_autofit_case():
    variogram = TheoreticalVariogram()
    variogram.autofit(experimental_variogram=ZEROS_VARIOGRAM, models_group='linear')
    assert variogram.rmse == 0
    assert variogram.nugget == 0


def test_we_direction_case():
    variogram = TheoreticalVariogram()
    variogram.autofit(experimental_variogram=WE_VARIOGRAM, models_group='all', nugget=0)

    expected_nugget = 0
    expected_sill = 4.25
    expected_range = 1.2

    assert variogram.nugget == expected_nugget
    assert np.allclose(variogram.sill, expected_sill, 2)
    assert np.allclose(variogram.rang, expected_range, 1)


def test_armstrong_case():
    variogram = TheoreticalVariogram()
    variogram.autofit(experimental_variogram=ARMSTRONG_VARIOGRAM, models_group='all', nugget=0)
    expected_sill = 12.85
    expected_range = 4.158
    assert np.allclose(variogram.sill, expected_sill, 2)
    assert np.allclose(variogram.rang, expected_range, 2)


def test_to_and_from_dict():
    variogram = TheoreticalVariogram()
    variogram.autofit(experimental_variogram=ZEROS_VARIOGRAM, models_group='linear')
    vdict = variogram.to_dict()

    variogram2 = TheoreticalVariogram()
    variogram2.from_dict(vdict)

    test = (
                   variogram.model_type == variogram2.model_type
           ) and (
                   variogram.sill == variogram2.sill
           ) and (
                   variogram.rang == variogram2.rang
           ) and (
                   variogram.nugget == variogram2.nugget
           )

    assert test


def test_to_and_from_json():
    fname = 'testfile'
    variogram = TheoreticalVariogram()
    variogram.autofit(experimental_variogram=ZEROS_VARIOGRAM, models_group='linear')
    variogram.to_json(fname)

    variogram2 = TheoreticalVariogram()
    variogram2.from_json(fname)

    test = (
                   variogram.model_type == variogram2.model_type
           ) and (
                   variogram.sill == variogram2.sill
           ) and (
                   variogram.rang == variogram2.rang
           ) and (
                   variogram.nugget == variogram2.nugget
           )

    # remove json
    os.remove(fname)

    assert test


def test_safe_autofit():
    variogram = TheoreticalVariogram()
    variogram.autofit(experimental_variogram=ZEROS_VARIOGRAM, models_group='safe')
    assert variogram


def test_nugget_autofit():
    variogram = TheoreticalVariogram()
    variogram.autofit(experimental_variogram=ARMSTRONG_VARIOGRAM, models_group='all')
    assert variogram.nugget > 0


def test__str__():
    variogram = TheoreticalVariogram()
    empty_str = "Theoretical model is not calculated yet. Use fit() or autofit() methods to build or find a model or import model with from_dict() or from_json() methods."
    assert variogram.__str__() == empty_str
    variogram.autofit(experimental_variogram=ARMSTRONG_VARIOGRAM,
                      models_group='linear')
    text = "* Selected model: Linear model"
    assert variogram.__str__().startswith(text)


def test__repr__():
    variogram = TheoreticalVariogram()
    empty_str = "Theoretical model is not calculated yet. Use fit() or autofit() methods to build or find a model or import model with from_dict() or from_json() methods."
    assert variogram.__repr__() == empty_str
    variogram.autofit(experimental_variogram=ARMSTRONG_VARIOGRAM,
                      models_group='linear')
    text = "* Selected model: Linear model"
    assert variogram.__repr__().startswith(text)
