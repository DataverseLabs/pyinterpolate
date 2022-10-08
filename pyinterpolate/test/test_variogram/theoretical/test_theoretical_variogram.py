import os
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
ARMSTRONG_VARIOGRAM = build_experimental_variogram(ARMSTRONG_DATA,
                                                   step_size=1,
                                                   max_range=6)
ARMSTRONG_VARIOGRAM_DIRECTIONAL = build_experimental_variogram(ARMSTRONG_DATA,
                                                               step_size=1.1,
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
        variogram = TheoreticalVariogram()
        variogram.autofit(experimental_variogram=ZEROS_VARIOGRAM, model_types='linear')
        self.assertEqual(0, variogram.rmse)
        self.assertEqual(0, variogram.nugget)

    def test_we_direction_case(self):
        variogram = TheoreticalVariogram()
        variogram.autofit(experimental_variogram=WE_VARIOGRAM, model_types='all')

        expected_nugget = 0
        expected_sill = 4.25
        expected_range = 1.2

        self.assertEqual(expected_nugget, variogram.nugget)
        self.assertAlmostEqual(expected_sill, variogram.sill, places=2)
        self.assertAlmostEqual(expected_range, variogram.rang, places=1)

    def test_armstrong_case(self):
        variogram = TheoreticalVariogram()
        variogram.autofit(experimental_variogram=ARMSTRONG_VARIOGRAM, model_types='all')
        expected_sill = 12.85
        expected_range = 4.13
        self.assertAlmostEqual(expected_sill, variogram.sill, places=2)
        self.assertAlmostEqual(expected_range, variogram.rang, places=2)

    def test_str_output(self):
        variogram = TheoreticalVariogram()
        output_str_empty = variogram.__str__()
        variogram.autofit(experimental_variogram=ZEROS_VARIOGRAM, model_types='linear')
        output_str_trained = variogram.__str__()
        expected_str_empty_model = 'Theoretical model is not calculated yet. ' \
                                   'Use fit() or autofit() methods to build or find a model ' \
                                   'or import model with from_dict() or from_json() methods.'
        expected_str_trained_model_startswith = '* Selected model: Linear model'

        msg_empty = 'Expected __str__() of the empty model is not equal to returned __str__().'
        self.assertEqual(output_str_empty, expected_str_empty_model, msg=msg_empty)

        msg_trained = 'Expected __str__() of trained model starts differently than the returned __str__().'
        self.assertTrue(output_str_trained.startswith(expected_str_trained_model_startswith), msg=msg_trained)

    def test_to_and_from_dict(self):
        variogram = TheoreticalVariogram()
        variogram.autofit(experimental_variogram=ZEROS_VARIOGRAM, model_types='linear')
        vdict = variogram.to_dict()

        variogram2 = TheoreticalVariogram()
        variogram2.from_dict(vdict)

        test = (
            variogram.name == variogram2.name
        ) and (
            variogram.sill == variogram2.sill
        ) and (
            variogram.rang == variogram2.rang
        ) and (
            variogram.nugget == variogram2.nugget
        )

        self.assertTrue(test, msg='Theoretical variograms should be the same! Saved and loaded dicts are different!')

    def test_to_and_from_json(self):
        fname = 'testfile'
        variogram = TheoreticalVariogram()
        variogram.autofit(experimental_variogram=ZEROS_VARIOGRAM, model_types='linear')
        variogram.to_json(fname)

        variogram2 = TheoreticalVariogram()
        variogram2.from_json(fname)

        test = (
                       variogram.name == variogram2.name
               ) and (
                       variogram.sill == variogram2.sill
               ) and (
                       variogram.rang == variogram2.rang
               ) and (
                       variogram.nugget == variogram2.nugget
               )

        # remove json
        os.remove(fname)

        self.assertTrue(test, msg='Theoretical variograms should be the same! Saved and loaded jsons are different!')
