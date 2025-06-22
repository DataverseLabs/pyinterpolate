from typing import Dict

import numpy as np

from pyinterpolate.kriging.point.indicator import IndicatorKriging
from pyinterpolate.semivariogram.indicator.indicator import ExperimentalIndicatorVariogram, \
    TheoreticalIndicatorVariogram


DEM = np.random.random(size=(1000, 3))

STEP_R = 0.1
MX_RNG = 0.6


# Divide data into training and test sets
def create_train_test(dataset: np.ndarray, training_set_ratio=0.3):
    """
    Function divides base dataset into a training and a test set.

    Parameters
    ----------
    dataset : np.ndarray

    training_set_ratio : float, default = 0.3

    Returns
    -------
    training_set, test_set : List[np.ndarray]
    """

    np.random.seed(101)  # To ensure that we will get the same results every time

    indexes_of_training_set = np.random.choice(range(len(dataset) - 1), int(training_set_ratio * len(dataset)),
                                               replace=False)

    training_set = dataset[indexes_of_training_set]
    validation_set = np.delete(dataset, indexes_of_training_set, 0)
    return training_set, validation_set


TRAIN, TEST = create_train_test(DEM)

EVARIOGRAM = ExperimentalIndicatorVariogram(ds=TRAIN,
                                            number_of_thresholds=3,
                                            step_size=0.1,
                                            max_range=0.6)

VARIOGRAMS = TheoreticalIndicatorVariogram(experimental_indicator_variogram=EVARIOGRAM)
VARIOGRAMS.fit()


def test_omni():
    ikriging = IndicatorKriging(
        known_locations=TRAIN,
        indicator_variograms=VARIOGRAMS,
        unknown_locations=TEST[:, :-1],
        kriging_type='ok',
        no_neighbors=16,
        allow_approximate_solutions=True
    )
    assert isinstance(ikriging, IndicatorKriging)
    assert isinstance(ikriging.indicator_predictions, np.ndarray)
    assert isinstance(ikriging.expected_values, np.ndarray)
    assert isinstance(ikriging.variances, np.ndarray)

    imaps = ikriging.get_indicator_maps()
    assert isinstance(imaps, Dict)
