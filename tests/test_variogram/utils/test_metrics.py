import unittest
import numpy as np

from pyinterpolate.variogram.utils.exceptions import UndefinedSMAPEWarning
from pyinterpolate.variogram.utils.metrics import root_mean_squared_error, weighted_root_mean_squared_error, \
    forecast_bias, symmetric_mean_absolute_percentage_error, mean_absolute_error


REAL0 = np.zeros(5)
PRED0 = np.zeros(5)

REAL_VALS = np.array([1, 2, 3, 3, 4])
PRED_VALS = REAL_VALS - 1

REAL_NEGATIVE = np.array([-3, 2, -6, 4, 3])
PRED_NEGATIVE = np.array([-4, 1, -4, 4, 1])

LAGS_VALS = np.array([30, 40, 20, 15, 5])


class TestMetrics(unittest.TestCase):

    def test_rmse(self):
        rmse0 = root_mean_squared_error(PRED0, REAL0)
        self.assertEqual(rmse0, 0)
        rmse_val = root_mean_squared_error(PRED_VALS, REAL_VALS)
        self.assertEqual(rmse_val, 1)
        rmse_with_negative = root_mean_squared_error(PRED_NEGATIVE, REAL_NEGATIVE)
        self.assertAlmostEqual(rmse_with_negative, 1.41, places=2)

    def test_weighted_rmse(self):
        # Dense
        method = 'dense'

        self.assertRaises(AttributeError, weighted_root_mean_squared_error, PRED0, REAL0, method)

        wrmse0 = weighted_root_mean_squared_error(PRED0, REAL0, method, LAGS_VALS)
        self.assertEqual(wrmse0, 0)

        wrmse_val = weighted_root_mean_squared_error(PRED_VALS, REAL_VALS, method, LAGS_VALS)
        self.assertAlmostEqual(wrmse_val, 0.447, places=3)

        wrmse_neg = weighted_root_mean_squared_error(PRED_NEGATIVE, REAL_NEGATIVE, method, LAGS_VALS)
        self.assertAlmostEqual(wrmse_neg, 0.556, places=3)

        # Closest
        method = 'closest'

        wrmse0 = weighted_root_mean_squared_error(PRED0, REAL0, method)
        self.assertEqual(wrmse0, 0)

        wrmse_val = weighted_root_mean_squared_error(PRED_VALS, REAL_VALS, method)
        self.assertAlmostEqual(wrmse_val, 0.632, places=3)

        wrmse_neg = weighted_root_mean_squared_error(PRED_NEGATIVE, REAL_NEGATIVE, method)
        self.assertAlmostEqual(wrmse_neg, 0.775, places=3)

        # Distant
        method = 'distant'

        wrmse0 = weighted_root_mean_squared_error(PRED0, REAL0, method)
        self.assertEqual(wrmse0, 0)

        wrmse_val = weighted_root_mean_squared_error(PRED_VALS, REAL_VALS, method)
        self.assertAlmostEqual(wrmse_val, 0.775, places=3)

        wrmse_neg = weighted_root_mean_squared_error(PRED_NEGATIVE, REAL_NEGATIVE, method)
        self.assertAlmostEqual(wrmse_neg, 1.183, places=3)

    def test_fb(self):
        fb0 = forecast_bias(PRED0, REAL0)
        self.assertEqual(fb0, 0)
        fb_val = forecast_bias(PRED_VALS, REAL_VALS)
        self.assertEqual(fb_val, 1)
        fb_with_negative = forecast_bias(PRED_NEGATIVE, REAL_NEGATIVE)
        self.assertAlmostEqual(fb_with_negative, 0.4, places=1)

    def test_smape(self):
        self.assertWarns(UndefinedSMAPEWarning, symmetric_mean_absolute_percentage_error, PRED0, REAL0)
        smape0 = symmetric_mean_absolute_percentage_error(PRED0, REAL0)
        self.assertEqual(smape0, 0)
        smape_val = symmetric_mean_absolute_percentage_error(PRED_VALS, REAL_VALS)
        self.assertAlmostEqual(smape_val, 37.5, places=1)
        smape_with_negative = symmetric_mean_absolute_percentage_error(PRED_NEGATIVE, REAL_NEGATIVE)
        self.assertAlmostEqual(smape_with_negative, 23.5, places=1)

    def test_mae(self):
        mae0 = mean_absolute_error(PRED0, REAL0)
        self.assertEqual(mae0, 0)
        mae_val = mean_absolute_error(PRED_VALS, REAL_VALS)
        self.assertEqual(mae_val, 1)
        mae_with_negative = mean_absolute_error(PRED_NEGATIVE, REAL_NEGATIVE)
        self.assertAlmostEqual(mae_with_negative, 1.2, places=1)
