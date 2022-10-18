# Core modules
import unittest
# Math modules
import numpy as np
# Tested module
from pyinterpolate.variogram.empirical.experimental_variogram import DirectionalVariogram
# Test data
from pyinterpolate.test.test_variogram.empirical.consts import get_armstrong_data


armstrong_arr = get_armstrong_data()
STEP_SIZE = 1
MAX_RANGE = 6


class TestDirectionalVariogramSelectionMethods(unittest.TestCase):

    def test_directions(self):
        dirvar_ellipsis = DirectionalVariogram(input_array=armstrong_arr,
                                               step_size=STEP_SIZE,
                                               max_range=MAX_RANGE,
                                               method='e',
                                               tolerance=0.3)

        dirvar_triangle = DirectionalVariogram(input_array=armstrong_arr,
                                               step_size=STEP_SIZE,
                                               max_range=MAX_RANGE,
                                               method='t',
                                               tolerance=0.7)

        # Test lags

        iso_e = dirvar_ellipsis.directional_variograms['ISO']
        iso_t = dirvar_triangle.directional_variograms['ISO']
        lags_e = iso_e.lags
        lags_t = iso_t.lags
        lags_equal = np.array_equal(lags_t, lags_e)
        self.assertTrue(lags_equal)

        # Test directional lags

        nesw_e = dirvar_ellipsis.directional_variograms['NE-SW']
        nesw_t = dirvar_triangle.directional_variograms['NE-SW']

        lags_e = nesw_e.lags
        lags_t = nesw_t.lags
        lags_equal = np.array_equal(lags_t, lags_e)
        self.assertTrue(lags_equal)

        # Test variances

        ns_t = dirvar_triangle.directional_variograms['NS']
        we_t = dirvar_triangle.directional_variograms['WE']
        nwse_t = dirvar_triangle.directional_variograms['NW-SE']

        nwse_e = dirvar_ellipsis.directional_variograms['NW-SE']
        ns_e = dirvar_ellipsis.directional_variograms['NS']
        we_e = dirvar_ellipsis.directional_variograms['WE']

        var_ns_e = ns_e.experimental_semivariances
        var_ns_t = ns_t.experimental_semivariances
        mean_ns_e = np.mean(var_ns_e)
        mean_ns_t = np.mean(var_ns_t)
        self.assertLess(np.abs(mean_ns_e - mean_ns_t), 3)

        var_we_e = we_e.experimental_semivariances
        var_we_t = we_t.experimental_semivariances
        mean_we_e = np.mean(var_we_e)
        mean_we_t = np.mean(var_we_t)
        self.assertLess(np.abs(mean_we_e - mean_we_t), 3)

        var_ne_sw_e = nesw_e.experimental_semivariances
        var_ne_sw_t = nesw_t.experimental_semivariances
        mean_ne_sw_e = np.mean(var_ne_sw_e)
        mean_ne_sw_t = np.mean(var_ne_sw_t)
        self.assertLess(np.abs(mean_ne_sw_e - mean_ne_sw_t), 3)

        var_nw_se_e = nwse_e.experimental_semivariances
        var_nw_se_t = nwse_t.experimental_semivariances
        mean_nw_se_e = np.mean(var_nw_se_e)
        mean_nw_se_t = np.mean(var_nw_se_t)
        self.assertLess(np.abs(mean_nw_se_e - mean_nw_se_t), 3)

        var_iso_e = iso_e.experimental_semivariances
        var_iso_t = iso_t.experimental_semivariances
        mean_iso_e = np.mean(var_iso_e)
        mean_iso_t = np.mean(var_iso_t)
        self.assertAlmostEqual(mean_iso_t, mean_iso_e, places=7)
