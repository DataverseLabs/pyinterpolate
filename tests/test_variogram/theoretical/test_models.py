import unittest
import numpy as np
from typing import Callable, Iterable

from pyinterpolate.variogram.theoretical.models import circular_model
from pyinterpolate.variogram.theoretical.models import cubic_model
from pyinterpolate.variogram.theoretical.models import exponential_model
from pyinterpolate.variogram.theoretical.models import gaussian_model
from pyinterpolate.variogram.theoretical.models import linear_model
from pyinterpolate.variogram.theoretical.models import power_model
from pyinterpolate.variogram.theoretical.models import spherical_model

from .consts import TheoreticalVariogramModelsData


VARIANCES = TheoreticalVariogramModelsData()
LAGS = VARIANCES.lags


def build_model(model_fn: Callable, lags: Iterable, nugget: float, sill: float, rang: float):
    mdl = model_fn(
        lags=lags,
        nugget=nugget,
        sill=sill,
        rang=rang
    )
    return mdl


class TestModels(unittest.TestCase):

    def test_circular(self):

        # n=0, s=0, r=0, l=(0:10)
        expected_variance_mean = 0
        cm_000010 = build_model(circular_model, LAGS, 0, 0, 0)
        modeled_variance_mean = np.mean(cm_000010)
        msg = 'The mean semivariance modeled for the parameters (n=0, s=0, r=0, l=(0:10)) should return 0'
        self.assertEqual(modeled_variance_mean, expected_variance_mean, msg=msg)

        # n=1, s=0, r=0, l=(0:10)
        expected_variance_mean = 1
        cm_100010 = build_model(circular_model, LAGS, 1, 0, 0)
        modeled_variance_mean = np.mean(cm_100010)
        msg = 'The mean semivariance modeled for the parameters (n=1, s=0, r=0, l=(0:10)) should return 1'
        self.assertEqual(modeled_variance_mean, expected_variance_mean, msg=msg)

        # n=1, s=1, r=1, l=(0:10)
        expected_lag0_value = 1
        expected_variance_mean = 1.9
        cm_111010 = build_model(circular_model, LAGS, 1, 1, 1)
        modeled_variance_mean = np.mean(cm_111010)
        msg_lag = 'Lag 0 value with nugget equal to 1 should be 1'
        msg_mean = 'The mean semivariance modeled for the parameters (n=1, s=1, r=1, l=(0:10)) should return 1.9'
        self.assertEqual(cm_111010[0], expected_lag0_value, msg_lag)
        self.assertEqual(modeled_variance_mean, expected_variance_mean, msg=msg_mean)

        # n=1, s=1, r=10, l=(0:10)
        expected_lag0_value = 1
        expected_variance_mean = 1.525
        cm_1110010 = build_model(circular_model, LAGS, 1, 1, 10)
        modeled_variance_mean = float(np.mean(cm_1110010))
        msg_mean = 'The mean semivariance modeled for the parameters (n=1, s=1, r=10, l=(0:10)) should return 1.525'
        self.assertEqual(cm_1110010[0], expected_lag0_value, msg_lag)
        self.assertAlmostEqual(modeled_variance_mean, expected_variance_mean, places=3, msg=msg_mean)

        # n=random, s=random, r=random, l=(0:10)
        rrange = VARIANCES.rang_random
        rnugget = VARIANCES.nugget_random
        rsill = VARIANCES.sill_random
        cm_random = build_model(circular_model,
                                LAGS,
                                rnugget,
                                rsill,
                                rrange)
        self.assertEqual(cm_random[0], rnugget, msg_lag)

    def test_cubic(self):
        # n=0, s=0, r=0, l=(0:10)
        expected_variance_mean = 0
        cm_000010 = build_model(cubic_model, LAGS, 0, 0, 0)
        modeled_variance_mean = np.mean(cm_000010)
        msg = 'The mean semivariance modeled for the parameters (n=0, s=0, r=0, l=(0:10)) should return 0'
        self.assertEqual(modeled_variance_mean, expected_variance_mean, msg=msg)

        # n=1, s=0, r=0, l=(0:10)
        expected_variance_mean = 1
        cm_100010 = build_model(cubic_model, LAGS, 1, 0, 0)
        modeled_variance_mean = np.mean(cm_100010)
        msg = 'The mean semivariance modeled for the parameters (n=1, s=0, r=0, l=(0:10)) should return 1'
        self.assertEqual(modeled_variance_mean, expected_variance_mean, msg=msg)

        # n=1, s=1, r=1, l=(0:10)
        expected_lag0_value = 1
        expected_variance_mean = 1.9
        cm_111010 = build_model(cubic_model, LAGS, 1, 1, 1)
        modeled_variance_mean = np.mean(cm_111010)
        msg_lag = 'Lag 0 value with nugget equal to 1 should be 1'
        msg_mean = 'The mean semivariance modeled for the parameters (n=1, s=1, r=1, l=(0:10)) should return 1.9'
        self.assertEqual(cm_111010[0], expected_lag0_value, msg_lag)
        self.assertEqual(modeled_variance_mean, expected_variance_mean, msg=msg_mean)

        # n=1, s=1, r=10, l=(0:10)
        expected_lag0_value = 1
        expected_variance_mean = 1.585
        cm_1110010 = build_model(cubic_model, LAGS, 1, 1, 10)
        modeled_variance_mean = float(np.mean(cm_1110010))
        msg_mean = 'The mean semivariance modeled for the parameters (n=1, s=1, r=10, l=(0:10)) should return 1.525'
        self.assertEqual(cm_1110010[0], expected_lag0_value, msg_lag)
        self.assertAlmostEqual(modeled_variance_mean, expected_variance_mean, places=3, msg=msg_mean)

        # n=random, s=random, r=random, l=(0:10)
        rrange = VARIANCES.rang_random
        rnugget = VARIANCES.nugget_random
        rsill = VARIANCES.sill_random
        cm_random = build_model(cubic_model,
                                LAGS,
                                rnugget,
                                rsill,
                                rrange)
        self.assertEqual(cm_random[0], rnugget, msg_lag)

    def test_exponential(self):
        # n=0, s=0, r=0, l=(0:10)
        expected_variance_mean = 0
        cm_000010 = build_model(exponential_model, LAGS, 0, 0, 0)
        modeled_variance_mean = np.mean(cm_000010)
        msg = 'The mean semivariance modeled for the parameters (n=0, s=0, r=0, l=(0:10)) should return 0'
        self.assertEqual(modeled_variance_mean, expected_variance_mean, msg=msg)

        # n=1, s=0, r=0, l=(0:10)
        expected_variance_mean = 1
        cm_100010 = build_model(exponential_model, LAGS, 1, 0, 0)
        modeled_variance_mean = np.mean(cm_100010)
        msg = 'The mean semivariance modeled for the parameters (n=1, s=0, r=0, l=(0:10)) should return 1'
        self.assertEqual(modeled_variance_mean, expected_variance_mean, msg=msg)

        # n=1, s=1, r=1, l=(0:10)
        expected_lag0_value = 1
        expected_variance_mean = 1.84
        cm_111010 = build_model(exponential_model, LAGS, 1, 1, 1)
        modeled_variance_mean = float(np.mean(cm_111010))
        msg_lag = 'Lag 0 value with nugget equal to 1 should be 1'
        msg_mean = 'The mean semivariance modeled for the parameters (n=1, s=1, r=1, l=(0:10)) should return 1.9'
        self.assertEqual(cm_111010[0], expected_lag0_value, msg_lag)
        self.assertAlmostEqual(modeled_variance_mean, expected_variance_mean, places=2, msg=msg_mean)

        # n=1, s=1, r=10, l=(0:10)
        expected_lag0_value = 1
        expected_variance_mean = 1.336
        cm_1110010 = build_model(exponential_model, LAGS, 1, 1, 10)
        modeled_variance_mean = float(np.mean(cm_1110010))
        msg_mean = 'The mean semivariance modeled for the parameters (n=1, s=1, r=10, l=(0:10)) should return 1.525'
        self.assertEqual(cm_1110010[0], expected_lag0_value, msg_lag)
        self.assertAlmostEqual(modeled_variance_mean, expected_variance_mean, places=3, msg=msg_mean)

        # n=random, s=random, r=random, l=(0:10)
        rrange = VARIANCES.rang_random
        rnugget = VARIANCES.nugget_random
        rsill = VARIANCES.sill_random
        cm_random = build_model(exponential_model,
                                LAGS,
                                rnugget,
                                rsill,
                                rrange)
        self.assertEqual(cm_random[0], rnugget, msg_lag)

    def test_gaussian(self):
        # n=0, s=0, r=0, l=(0:10)
        expected_variance_mean = 0
        cm_000010 = build_model(gaussian_model, LAGS, 0, 0, 0)
        modeled_variance_mean = np.mean(cm_000010)
        msg = 'The mean semivariance modeled for the parameters (n=0, s=0, r=0, l=(0:10)) should return 0'
        self.assertEqual(modeled_variance_mean, expected_variance_mean, msg=msg)

        # n=1, s=0, r=0, l=(0:10)
        expected_variance_mean = 1
        cm_100010 = build_model(gaussian_model, LAGS, 1, 0, 0)
        modeled_variance_mean = np.mean(cm_100010)
        msg = 'The mean semivariance modeled for the parameters (n=1, s=0, r=0, l=(0:10)) should return 1'
        self.assertEqual(modeled_variance_mean, expected_variance_mean, msg=msg)

        # n=1, s=1, r=1, l=(0:10)
        expected_lag0_value = 1
        expected_variance_mean = 1.86
        cm_111010 = build_model(gaussian_model, LAGS, 1, 1, 1)
        modeled_variance_mean = float(np.mean(cm_111010))
        msg_lag = 'Lag 0 value with nugget equal to 1 should be 1'
        msg_mean = 'The mean semivariance modeled for the parameters (n=1, s=1, r=1, l=(0:10)) should return 1.9'
        self.assertEqual(cm_111010[0], expected_lag0_value, msg_lag)
        self.assertAlmostEqual(modeled_variance_mean, expected_variance_mean, places=2, msg=msg_mean)

        # n=1, s=1, r=10, l=(0:10)
        expected_lag0_value = 1
        expected_variance_mean = 1.222
        cm_1110010 = build_model(gaussian_model, LAGS, 1, 1, 10)
        modeled_variance_mean = float(np.mean(cm_1110010))
        msg_mean = 'The mean semivariance modeled for the parameters (n=1, s=1, r=10, l=(0:10)) should return 1.525'
        self.assertEqual(cm_1110010[0], expected_lag0_value, msg_lag)
        self.assertAlmostEqual(modeled_variance_mean, expected_variance_mean, places=3, msg=msg_mean)

        # n=random, s=random, r=random, l=(0:10)
        rrange = VARIANCES.rang_random
        rnugget = VARIANCES.nugget_random
        rsill = VARIANCES.sill_random
        cm_random = build_model(gaussian_model,
                                LAGS,
                                rnugget,
                                rsill,
                                rrange)
        self.assertEqual(cm_random[0], rnugget, msg_lag)

    def test_linear(self):
        # n=0, s=0, r=0, l=(0:10)
        expected_variance_mean = 0
        cm_000010 = build_model(linear_model, LAGS, 0, 0, 0)
        modeled_variance_mean = np.mean(cm_000010)
        msg = 'The mean semivariance modeled for the parameters (n=0, s=0, r=0, l=(0:10)) should return 0'
        self.assertEqual(modeled_variance_mean, expected_variance_mean, msg=msg)

        # n=1, s=0, r=0, l=(0:10)
        expected_variance_mean = 1
        cm_100010 = build_model(linear_model, LAGS, 1, 0, 0)
        modeled_variance_mean = np.mean(cm_100010)
        msg = 'The mean semivariance modeled for the parameters (n=1, s=0, r=0, l=(0:10)) should return 1'
        self.assertEqual(modeled_variance_mean, expected_variance_mean, msg=msg)

        # n=1, s=1, r=1, l=(0:10)
        expected_lag0_value = 1
        expected_variance_mean = 1.9
        cm_111010 = build_model(linear_model, LAGS, 1, 1, 1)
        modeled_variance_mean = float(np.mean(cm_111010))
        msg_lag = 'Lag 0 value with nugget equal to 1 should be 1'
        msg_mean = 'The mean semivariance modeled for the parameters (n=1, s=1, r=1, l=(0:10)) should return 1.9'
        self.assertEqual(cm_111010[0], expected_lag0_value, msg_lag)
        self.assertAlmostEqual(modeled_variance_mean, expected_variance_mean, places=2, msg=msg_mean)

        # n=1, s=1, r=10, l=(0:10)
        expected_lag0_value = 1
        expected_variance_mean = 1.45
        cm_1110010 = build_model(linear_model, LAGS, 1, 1, 10)
        modeled_variance_mean = float(np.mean(cm_1110010))
        msg_mean = 'The mean semivariance modeled for the parameters (n=1, s=1, r=10, l=(0:10)) should return 1.525'
        self.assertEqual(cm_1110010[0], expected_lag0_value, msg_lag)
        self.assertAlmostEqual(modeled_variance_mean, expected_variance_mean, places=3, msg=msg_mean)

        # n=random, s=random, r=random, l=(0:10)
        rrange = VARIANCES.rang_random
        rnugget = VARIANCES.nugget_random
        rsill = VARIANCES.sill_random
        cm_random = build_model(linear_model,
                                LAGS,
                                rnugget,
                                rsill,
                                rrange)
        self.assertEqual(cm_random[0], rnugget, msg_lag)

    def test_power(self):
        # n=0, s=0, r=0, l=(0:10)
        expected_variance_mean = 0
        cm_000010 = build_model(power_model, LAGS, 0, 0, 0)
        modeled_variance_mean = np.mean(cm_000010)
        msg = 'The mean semivariance modeled for the parameters (n=0, s=0, r=0, l=(0:10)) should return 0'
        self.assertEqual(modeled_variance_mean, expected_variance_mean, msg=msg)

        # n=1, s=0, r=0, l=(0:10)
        expected_variance_mean = 1
        cm_100010 = build_model(power_model, LAGS, 1, 0, 0)
        modeled_variance_mean = np.mean(cm_100010)
        msg = 'The mean semivariance modeled for the parameters (n=1, s=0, r=0, l=(0:10)) should return 1'
        self.assertEqual(modeled_variance_mean, expected_variance_mean, msg=msg)

        # n=1, s=1, r=1, l=(0:10)
        expected_lag0_value = 1
        expected_variance_mean = 1.9
        cm_111010 = build_model(power_model, LAGS, 1, 1, 1)
        modeled_variance_mean = float(np.mean(cm_111010))
        msg_lag = 'Lag 0 value with nugget equal to 1 should be 1'
        msg_mean = 'The mean semivariance modeled for the parameters (n=1, s=1, r=1, l=(0:10)) should return 1.9'
        self.assertEqual(cm_111010[0], expected_lag0_value, msg_lag)
        self.assertAlmostEqual(modeled_variance_mean, expected_variance_mean, places=2, msg=msg_mean)

        # n=1, s=1, r=10, l=(0:10)
        expected_lag0_value = 1
        expected_variance_mean = 1.285
        cm_1110010 = build_model(power_model, LAGS, 1, 1, 10)
        modeled_variance_mean = float(np.mean(cm_1110010))
        msg_mean = 'The mean semivariance modeled for the parameters (n=1, s=1, r=10, l=(0:10)) should return 1.525'
        self.assertEqual(cm_1110010[0], expected_lag0_value, msg_lag)
        self.assertAlmostEqual(modeled_variance_mean, expected_variance_mean, places=3, msg=msg_mean)

        # n=random, s=random, r=random, l=(0:10)
        rrange = VARIANCES.rang_random
        rnugget = VARIANCES.nugget_random
        rsill = VARIANCES.sill_random
        cm_random = build_model(power_model,
                                LAGS,
                                rnugget,
                                rsill,
                                rrange)
        self.assertEqual(cm_random[0], rnugget, msg_lag)

    def test_spherical(self):
        # n=0, s=0, r=0, l=(0:10)
        expected_variance_mean = 0
        cm_000010 = build_model(spherical_model, LAGS, 0, 0, 0)
        modeled_variance_mean = np.mean(cm_000010)
        msg = 'The mean semivariance modeled for the parameters (n=0, s=0, r=0, l=(0:10)) should return 0'
        self.assertEqual(modeled_variance_mean, expected_variance_mean, msg=msg)

        # n=1, s=0, r=0, l=(0:10)
        expected_variance_mean = 1
        cm_100010 = build_model(spherical_model, LAGS, 1, 0, 0)
        modeled_variance_mean = np.mean(cm_100010)
        msg = 'The mean semivariance modeled for the parameters (n=1, s=0, r=0, l=(0:10)) should return 1'
        self.assertEqual(modeled_variance_mean, expected_variance_mean, msg=msg)

        # n=1, s=1, r=1, l=(0:10)
        expected_lag0_value = 1
        expected_variance_mean = 1.9
        cm_111010 = build_model(spherical_model, LAGS, 1, 1, 1)
        modeled_variance_mean = float(np.mean(cm_111010))
        msg_lag = 'Lag 0 value with nugget equal to 1 should be 1'
        msg_mean = 'The mean semivariance modeled for the parameters (n=1, s=1, r=1, l=(0:10)) should return 1.9'
        self.assertEqual(cm_111010[0], expected_lag0_value, msg_lag)
        self.assertAlmostEqual(modeled_variance_mean, expected_variance_mean, places=2, msg=msg_mean)

        # n=1, s=1, r=10, l=(0:10)
        expected_lag0_value = 1
        expected_variance_mean = 1.57375
        cm_1110010 = build_model(spherical_model, LAGS, 1, 1, 10)
        modeled_variance_mean = float(np.mean(cm_1110010))
        msg_mean = 'The mean semivariance modeled for the parameters (n=1, s=1, r=10, l=(0:10)) should return 1.525'
        self.assertEqual(cm_1110010[0], expected_lag0_value, msg_lag)
        self.assertAlmostEqual(modeled_variance_mean, expected_variance_mean, places=3, msg=msg_mean)

        # n=random, s=random, r=random, l=(0:10)
        rrange = VARIANCES.rang_random
        rnugget = VARIANCES.nugget_random
        rsill = VARIANCES.sill_random
        cm_random = build_model(spherical_model,
                                LAGS,
                                rnugget,
                                rsill,
                                rrange)
        self.assertEqual(cm_random[0], rnugget, msg_lag)
