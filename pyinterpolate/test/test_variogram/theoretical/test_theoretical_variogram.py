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
ARMSTRONG_VARIOGRAM = build_experimental_variogram(ARMSTRONG_DATA,
                                                   step_size=1,
                                                   max_range=6)
ARMSTRONG_VARIOGRAM_DIRECTIONAL = build_experimental_variogram(ARMSTRONG_DATA,
                                                               step_size=1,
                                                               max_range=6,
                                                               direction=135,
                                                               tolerance=0.02)

class TestTheoreticalVariogram(unittest.TestCase):

    def test_zero_case(self):
        _sill = 10
        _range = 10
        _theo = build_theoretical_variogram(
            ZEROS_VARIOGRAM, 'linear', _sill, _range
        )

        expected_bias = -3
        expected_rmse = 3.32
        self.assertEqual(expected_bias, _theo.bias)
        self.assertAlmostEqual(expected_rmse, _theo.rmse, places=2)

    def test_zero_autofit_case(self):
        variogram = TheoreticalVariogram(ZEROS_VARIOGRAM)
        variogram.autofit(model_types='linear')
        self.assertEqual(0, variogram.rmse)
        self.assertEqual(0, variogram.nugget)

    def test_we_direction_case(self):
        variogram = TheoreticalVariogram(WE_VARIOGRAM)
        variogram.autofit(model_types='all')

        expected_nugget = 0
        expected_sill = 4.25
        expected_range = 1.2

        self.assertEqual(expected_nugget, variogram.nugget)
        self.assertAlmostEqual(expected_sill, variogram.sill, places=2)
        self.assertAlmostEqual(expected_range, variogram.rang, places=1)

    # def test_weighted_case(self):
        # variogram = TheoreticalVariogram(WEIGHTED_VARIOGRAM)
        # variogram.autofit(model_types='all')
        #
        # print(variogram)

    def test_armstrong_case(self):
        variogram = TheoreticalVariogram(ARMSTRONG_VARIOGRAM)
        variogram.autofit(model_types='all')
        expected_sill = 12.85
        expected_range = 4.16
        self.assertAlmostEqual(expected_sill, variogram.sill, places=2)
        self.assertAlmostEqual(expected_range, variogram.rang, places=2)

    def test_str_output(self):
        pass

    def test_repr_eval(self):
        pass