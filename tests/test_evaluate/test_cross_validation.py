import os

import numpy as np

from pyinterpolate.semivariogram.experimental.classes.experimental_variogram import ExperimentalVariogram
from pyinterpolate.evaluate.cross_validation import validate_kriging
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram

from tests.test_semivariogram._ds import get_armstrong_data


ARMSTRONG_DATA = get_armstrong_data()
SIMPLE_MEAN = np.mean(ARMSTRONG_DATA[:, -1])
ARMSTRONG_VARIOGRAM = ExperimentalVariogram(ARMSTRONG_DATA,
                                            step_size=1,
                                            max_range=6)
THEORETICAL_MODEL = TheoreticalVariogram()
THEORETICAL_MODEL.autofit(experimental_variogram=ARMSTRONG_VARIOGRAM, models_group='linear', nugget=0.0)


def test_with_ordinary():
    validation_results = validate_kriging(
        ARMSTRONG_DATA, theoretical_model=THEORETICAL_MODEL, no_neighbors=4
    )

    # Number of points is the same as in input data
    assert len(validation_results[2]) == len(ARMSTRONG_DATA)

    # Average error ~ (-0.016)
    assert abs(validation_results[0] - 0.016) < 0.1

    # Average variance error ~ 1.63
    assert validation_results[1] > 1.60
    assert validation_results[1] < 1.65

def test_with_simple():
    validation_results = validate_kriging(
        ARMSTRONG_DATA,
        theoretical_model=THEORETICAL_MODEL,
        no_neighbors=4,
        how='sk',
        sk_mean=SIMPLE_MEAN
    )

    # Number of points is the same as in input data
    assert len(validation_results[2]) == len(ARMSTRONG_DATA)

    # Average error ~ (-0.015)
    assert abs(validation_results[0] - 0.016) < 0.1

    # Average variance error ~ 1.58
    assert validation_results[1] > 1.55
    assert validation_results[1] < 1.60
