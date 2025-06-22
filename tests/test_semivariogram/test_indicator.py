from typing import Dict

import numpy as np

from pyinterpolate.semivariogram.indicator.indicator import ExperimentalIndicatorVariogram, IndicatorVariogramData, \
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


def test_experimental_indicator_variogram():
    evar = ExperimentalIndicatorVariogram(ds=TRAIN,
                                          number_of_thresholds=5,
                                          step_size=STEP_R,
                                          max_range=MX_RNG)

    # evar.show()

    assert isinstance(evar, ExperimentalIndicatorVariogram)
    assert isinstance(evar.experimental_models, Dict)
    assert isinstance(evar.ds, IndicatorVariogramData)


def test_theoretical_indicator_variogram():
    evar = ExperimentalIndicatorVariogram(ds=TRAIN,
                                          number_of_thresholds=5,
                                          step_size=STEP_R,
                                          max_range=MX_RNG)
    vrgs = TheoreticalIndicatorVariogram(
        experimental_indicator_variogram=evar
    )
    vrgs.fit()
    # vrgs.show()
    assert isinstance(vrgs, TheoreticalIndicatorVariogram)
    assert isinstance(vrgs.theoretical_indicator_variograms, Dict)
    assert len(vrgs.theoretical_indicator_variograms) == 5
