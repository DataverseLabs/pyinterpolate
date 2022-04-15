# Python core
from typing import List, Union, Tuple

# Core calculation and data visualization
import numpy as np

# Pyinterpolate
from pyinterpolate.kriging.utils.matrices import get_predictions
from pyinterpolate.variogram import TheoreticalVariogram


def simple_kriging(
        theoretical_model: TheoreticalVariogram,
        known_locations: np.ndarray,
        unknown_location: Union[List, Tuple, np.ndarray],
        process_mean: float,
        neighbors_range=None,
        min_no_neighbors=1,
        max_no_neighbors=-1
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

    Returns
    -------
    : numpy array
        [predicted value, variance error, longitude (x), latitude (y)]

    Warns
    -----
    TODO
    NegativeWeightsWarning : set if weights in weighting matrix are negative.

    """

    k, predicted, dataset = get_predictions(theoretical_model,
                                            known_locations,
                                            unknown_location,
                                            neighbors_range,
                                            min_no_neighbors,
                                            max_no_neighbors)

    w = np.linalg.solve(predicted, k)
    r = dataset[:, -2] - process_mean
    zhat = r.dot(w)
    zhat = zhat + process_mean

    sigma = np.matmul(w.T, k)

    return [zhat, sigma, unknown_location[0], unknown_location[1]]
