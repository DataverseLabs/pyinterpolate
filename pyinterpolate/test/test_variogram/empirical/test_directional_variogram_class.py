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


class TestDirectionalVariogram(unittest.TestCase):

    def test_class_init(self):
        dirvar = DirectionalVariogram(input_array=armstrong_arr,
                                      step_size=STEP_SIZE,
                                      max_range=MAX_RANGE)
        self.assertTrue(bool(dirvar.directional_variograms))

    def test_directions(self):
        dirvar = DirectionalVariogram(input_array=armstrong_arr,
                                      step_size=STEP_SIZE,
                                      max_range=MAX_RANGE,
                                      method='e')
        iso = dirvar.directional_variograms['ISO']
        ns = dirvar.directional_variograms['NS']
        we = dirvar.directional_variograms['WE']
        nesw = dirvar.directional_variograms['NE-SW']
        nwse = dirvar.directional_variograms['NW-SE']

        the_same_lags = np.array_equal(np.mean([iso.lags,
                                                ns.lags,
                                                we.lags,
                                                nesw.lags,
                                                nwse.lags], axis=0), iso.lags)

        self.assertTrue(the_same_lags)

        iso_output = [5.7, 8.4, 11.1, 12.1, 15.0]
        we_output = [5.0, 8.8, 8.5, 12.8, 13.0]
        ns_output = [6.4, 9.5, 8.1, 10.0, 10.7]
        nwse_output = [0, 6.2, 11.0, 10.3, 11.8]
        nesw_output = [0, 6.3, 11.5, 9.7, 10.7]

        isovar = iso.experimental_semivariances
        nsvar = ns.experimental_semivariances
        wevar = we.experimental_semivariances
        neswvar = nesw.experimental_semivariances
        nwsevar = nwse.experimental_semivariances

        self.assertTrue(np.allclose(isovar, iso_output, rtol=0.1))
        self.assertTrue(np.allclose(nsvar, ns_output, rtol=0.1))
        self.assertTrue(np.allclose(wevar, we_output, rtol=0.1))
        self.assertTrue(np.allclose(neswvar, nesw_output, rtol=0.1))
        self.assertTrue(np.allclose(nwsevar, nwse_output, rtol=0.1))
