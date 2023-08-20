import unittest
from typing import Dict

import numpy as np

from pyinterpolate import IndicatorVariogramData
from pyinterpolate import read_txt, ExperimentalIndicatorVariogram, IndicatorVariograms

DEM = read_txt('samples/point_data/txt/pl_dem_epsg2180.txt')

STEP_R = 500  # meters
MX_RNG = 10000  # meters


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


class TestExperimentalIndicatorVariogram(unittest.TestCase):

    def test_flow_1(self):
        evar = ExperimentalIndicatorVariogram(input_array=TRAIN,
                                              number_of_thresholds=5,
                                              step_size=STEP_R,
                                              max_range=MX_RNG)
        self.assertIsInstance(evar, ExperimentalIndicatorVariogram)
        self.assertIsInstance(evar.experimental_models, Dict)
        self.assertIsInstance(evar.ds, IndicatorVariogramData)


class TestIndicatorVariograms(unittest.TestCase):

    def test_flow_1(self):
        evar = ExperimentalIndicatorVariogram(input_array=TRAIN,
                                              number_of_thresholds=5,
                                              step_size=STEP_R,
                                              max_range=MX_RNG)
        variograms = IndicatorVariograms(experimental_indicator_variogram=evar)
        variograms.fit(
            model_name='safe',
            verbose=False
        )
        self.assertIsInstance(variograms, IndicatorVariograms)
