import unittest
import numpy as np

from pyinterpolate.variogram import build_experimental_variogram, TheoreticalVariogram
from pyinterpolate.test.test_variogram.theoretical.consts import TheoreticalVariogramTestData
from pyinterpolate.variogram.regularization.deconvolution import calculate_deviation

VARIOGRAM_DATA = TheoreticalVariogramTestData()

WE_VARIOGRAM = build_experimental_variogram(VARIOGRAM_DATA.input_data_we,
                                            step_size=VARIOGRAM_DATA.param_step_size,
                                            max_range=VARIOGRAM_DATA.param_max_range)

VARIOGRAM = TheoreticalVariogram()
VARIOGRAM.autofit(experimental_variogram=WE_VARIOGRAM, model_types='all')


class TestCalculateDeviation(unittest.TestCase):

    def test_random_case(self):
        mask_arr = WE_VARIOGRAM.experimental_semivariance_array.copy()
        input_values = np.random.random(len(mask_arr))
        mask_arr[:, 1] = input_values
        mask_arr = mask_arr[:, :-1]
        deviation = calculate_deviation(VARIOGRAM, mask_arr)
        self.assertTrue(deviation)

    def test_zeros_case(self):
        mask_arr = WE_VARIOGRAM.experimental_semivariance_array.copy()
        mask_arr[0, 1] = 1
        mask_arr[1:, 1] = 0
        mask_arr = mask_arr[:, :-1]
        deviation = calculate_deviation(VARIOGRAM, mask_arr)
        self.assertTrue(deviation)
