import unittest
from typing import Dict

import numpy as np

from pyinterpolate import read_txt, ExperimentalIndicatorVariogram, IndicatorVariograms, IndicatorKriging


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

EVARIOGRAM = ExperimentalIndicatorVariogram(input_array=TRAIN,
                                            number_of_thresholds=5,
                                            step_size=STEP_R,
                                            max_range=MX_RNG)

VARIOGRAMS = IndicatorVariograms(experimental_indicator_variogram=EVARIOGRAM)
VARIOGRAMS.fit(
    model_type='basic',
    verbose=False
)

class TestIndicatorKriging(unittest.TestCase):

    def test_flow_1(self):
        ikriging = IndicatorKriging(
            datapoints=TRAIN,
            indicator_variograms=VARIOGRAMS,
            unknown_locations=TEST[:, :-1],
            kriging_type='ok',
            no_neighbors=4,
            allow_approximate_solutions=True
        )
        self.assertIsInstance(ikriging, IndicatorKriging)
        self.assertIsInstance(ikriging.indicator_predictions, np.ndarray)
        self.assertIsInstance(ikriging.expected_values, np.ndarray)
        self.assertIsInstance(ikriging.variances, np.ndarray)

        imaps = ikriging.get_indicator_maps()
        self.assertIsInstance(imaps, Dict)
