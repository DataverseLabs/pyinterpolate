# TODO: log k, predicted, dataset
# Python core
from typing import List, Union, Tuple

# Core calculation and data visualization
import numpy as np

# Pyinterpolate
from pyinterpolate.kriging.utils.process import get_predictions, solve_weights
from pyinterpolate.variogram import TheoreticalVariogram


def simple_kriging(
        theoretical_model: TheoreticalVariogram,
        known_locations: np.ndarray,
        unknown_location: Union[List, Tuple, np.ndarray],
        process_mean: float,
        neighbors_range=None,
        min_no_neighbors=1,
        max_no_neighbors=-1,
        allow_approximate_solutions=False
) -> List:
    """
    Function predicts value at unknown location with Ordinary Kriging technique.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
                        Trained theoretical variogram model.

    known_locations : numpy array
                      Array with the known locations.

    unknown_location : Union[List, Tuple, numpy array]
                       Point where you want to estimate value (x, y) <-> (lon, lat)

    process_mean : float
                   The mean value of a process over a study area. Should be know before processing. That's why Simple
                   Kriging has limited number of applications. You must have multiple samples and well-known area to
                   know this parameter.

    neighbors_range : float, default = None
                      Maximum distance where we search for point neighbors. If None given then range is selected from
                      the theoretical_model rang attribute.

    min_no_neighbors : int, default = 1
                       Minimum number of neighbors to estimate unknown value; value is used when insufficient number of
                       neighbors is within neighbors_range.

    max_no_neighbors : int, default = -1
                       Maximum number of n-closest neighbors used for interpolation if there are too many neighbors
                       in neighbors_range. It speeds up calculations for large datasets. Default -1 means that
                       all possible neighbors will be used.

    allow_approximate_solutions : bool, default = False
                                  Allows the approximation of kriging weights based on the OLS algorithm.
                                  Not recommended to set to True if you don't know what you are doing!

    Returns
    -------
    : numpy array
        [predicted value, variance error, longitude (x), latitude (y)]

    """

    k, predicted, dataset = get_predictions(theoretical_model,
                                            known_locations,
                                            unknown_location,
                                            neighbors_range,
                                            min_no_neighbors,
                                            max_no_neighbors)

    try:
        output_weights = solve_weights(predicted, k, allow_approximate_solutions)
    except np.linalg.LinAlgError as err_numpy:
        # TODO: log k, predicted, dataset
        return [np.nan, np.nan, unknown_location[0], unknown_location[1]]

    r = dataset[:, -2] - process_mean
    zhat = r.dot(output_weights)
    zhat = zhat + process_mean

    sigma = np.matmul(output_weights.T, k)

    return [zhat, sigma, unknown_location[0], unknown_location[1]]
