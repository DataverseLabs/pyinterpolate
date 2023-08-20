import unittest
import numpy as np

from pyinterpolate import build_experimental_variogram, TheoreticalVariogram
from pyinterpolate.validation.cross_validation import validate_kriging

from tests.test_variogram.empirical.consts import get_armstrong_data


ARMSTRONG_DATA = get_armstrong_data()
SIMPLE_MEAN = np.mean(ARMSTRONG_DATA[:, -1])
ARMSTRONG_VARIOGRAM = build_experimental_variogram(ARMSTRONG_DATA,
                                                   step_size=1,
                                                   max_range=6)
THEORETICAL_MODEL = TheoreticalVariogram()
THEORETICAL_MODEL.autofit(experimental_variogram=ARMSTRONG_VARIOGRAM, model_name='linear', nugget=0.0)


class TestValidateKriging(unittest.TestCase):

    def test_with_ordinary(self):
        validation_results = validate_kriging(
            ARMSTRONG_DATA, theoretical_model=THEORETICAL_MODEL, no_neighbors=4
        )

        # Number of points is the same as in input data
        self.assertEqual(len(validation_results[2]), len(ARMSTRONG_DATA))

        # Average error ~ (-0.016)
        self.assertAlmostEqual(validation_results[0], -0.016, places=3)

        # Average variance error ~ 1.67
        self.assertAlmostEqual(validation_results[1], 1.729, places=2)

    def test_with_simple(self):
        validation_results = validate_kriging(
            ARMSTRONG_DATA,
            theoretical_model=THEORETICAL_MODEL,
            no_neighbors=4,
            how='sk',
            sk_mean=SIMPLE_MEAN
        )

        # Number of points is the same as in input data
        self.assertEqual(len(validation_results[2]), len(ARMSTRONG_DATA))

        # Average error ~ (-0.015)
        self.assertAlmostEqual(validation_results[0], -0.015, places=3)

        # Average variance error ~ 1.67
        self.assertAlmostEqual(validation_results[1], 1.67, places=2)
