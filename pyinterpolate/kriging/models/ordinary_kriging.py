# Python core
from typing import List, Union, Tuple

# Core calculation and data visualization
import numpy as np

# Pyinterpolate
from pyinterpolate.kriging.utils.matrices import get_predictions, solve_weights
from pyinterpolate.variogram import TheoreticalVariogram


def ordinary_kriging(
        theoretical_model: TheoreticalVariogram,
        known_locations: np.ndarray,
        unknown_location: Union[List, Tuple, np.ndarray],
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

    k_ones = np.ones(1)[0]
    k = np.r_[k, k_ones]

    p_ones = np.ones((predicted.shape[0], 1))
    predicted_with_ones_col = np.c_[predicted, p_ones]
    p_ones_row = np.ones((1, predicted_with_ones_col.shape[1]))
    p_ones_row[0][-1] = 0.
    weights = np.r_[predicted_with_ones_col, p_ones_row]

    output_weights = solve_weights(weights, k, allow_approximate_solutions)
    try:
        zhat = dataset[:, -2].dot(output_weights[:-1])
    except ValueError:
        print(dataset)

    sigma = np.matmul(output_weights.T, k)

    return [zhat, sigma, unknown_location[0], unknown_location[1]]
