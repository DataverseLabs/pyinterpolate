import unittest
from pyinterpolate.variogram import build_experimental_variogram
from pyinterpolate.variogram import build_theoretical_variogram, TheoreticalVariogram
from pyinterpolate.test.test_variogram.theoretical.consts import TheoreticalVariogramTestData
from pyinterpolate.test.test_variogram.theoretical.consts import get_armstrong_data


VARIOGRAM_DATA = TheoreticalVariogramTestData()
ARMSTRONG_DATA = get_armstrong_data()
ZEROS_VARIOGRAM = build_experimental_variogram(VARIOGRAM_DATA.input_zeros,
                                               step_size=VARIOGRAM_DATA.param_step_size,
                                               max_range=VARIOGRAM_DATA.param_max_range)
WE_VARIOGRAM = build_experimental_variogram(VARIOGRAM_DATA.input_data_we,
                                            step_size=VARIOGRAM_DATA.param_step_size,
                                            max_range=VARIOGRAM_DATA.param_max_range)
WEIGHTED_VARIOGRAM = build_experimental_variogram(VARIOGRAM_DATA.input_weighted[0],
                                                  step_size=VARIOGRAM_DATA.param_step_size,
                                                  max_range=VARIOGRAM_DATA.param_max_range,
                                                  weights=VARIOGRAM_DATA.input_weighted[1][:, -1])
ARMSTRONG_VARIOGRAM_DIRECTIONAL = build_experimental_variogram(ARMSTRONG_DATA,
                                                               step_size=1,
                                                               max_range=6,
                                                               direction=135,
                                                               tolerance=0.02)

class TestTheoreticalVariogram:

    def test_zero_case(self):
        _sill = 10
        _range = 10
        _theo = build_theoretical_variogram(
            ZEROS_VARIOGRAM, 'linear', _sill, _range
        )

        print(_theo)

    def test_we_direction_case(self):
        pass

    def test_weighted_case(self):
        pass

    def test_directional_case(self):
        pass

    def test_str_output(self):
        pass

    def test_repr_eval(self):
        pass