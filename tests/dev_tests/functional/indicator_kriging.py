import matplotlib.pyplot as plt
import numpy as np

from pyinterpolate import read_txt
from pyinterpolate.variogram.indicator.indicator_variogram import ExperimentalIndicatorVariogram, IndicatorVariograms
from pyinterpolate.kriging.models.indicator.indicator_point_kriging import IndicatorKriging

dem = read_txt('../../samples/point_data/txt/pl_dem_epsg2180.txt')


step_radius = 500  # meters
_max_range = 10000  # meters

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


train_set, test_set = create_train_test(dem)

ind_variogram = ExperimentalIndicatorVariogram(input_array=train_set,
                                               number_of_thresholds=20,
                                               step_size=step_radius,
                                               max_range=_max_range)

ind_vars = IndicatorVariograms(experimental_indicator_variogram=ind_variogram)
ind_vars.fit(
    model_type='basic',
    verbose=False
)

ikriging = IndicatorKriging(
    datapoints=train_set,
    indicator_variograms=ind_vars,
    unknown_locations=test_set[:, :-1],
    kriging_type='ok',
    no_neighbors=25,
    allow_approximate_solutions=True
)

inds_predictions = ikriging.indicator_predictions
expected_values = ikriging.expected_values
