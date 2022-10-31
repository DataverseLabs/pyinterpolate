"""
Solves kriging weights and gets per-distance weights from a semivariogram model.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
import warnings
from typing import List, Union, Tuple

import numpy as np

from pyinterpolate.distance.distance import calc_point_to_point_distance
from pyinterpolate.kriging.utils.kwarnings import ZerosMatrixWarning, LeastSquaresApproximationWarning
from pyinterpolate.processing.select_values import select_kriging_data, select_kriging_data_from_direction
from pyinterpolate.variogram import TheoreticalVariogram


def get_predictions(theoretical_model: TheoreticalVariogram,
                    known_locations: np.ndarray,
                    unknown_location: Union[List, Tuple, np.ndarray],
                    neighbors_range=None,
                    no_neighbors=4,
                    use_all_neighbors_in_range=False) -> List:
    """
    Function predicts semivariances for distances between points and unknown points, and between known points and
    returns two predicted arrays.

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

    no_neighbors : int, default = 4
                   Number of the n-closest neighbors used for interpolation.

    use_all_neighbors_in_range : bool, default = False
                                 True: if number of neighbors within the neighbors_range is greater than the
                                 number_of_neighbors then take all of them for modeling.

    Returns
    -------
    : List[predictions - unknown point, predictions - point to point, prepared Kriging data]
    """

    # Check range
    if neighbors_range is None:
        neighbors_range = theoretical_model.rang

    if theoretical_model.direction is not None:
        prepared_data = select_kriging_data_from_direction(unknown_position=unknown_location,
                                                           data_array=known_locations,
                                                           neighbors_range=neighbors_range,
                                                           direction=theoretical_model.direction,
                                                           number_of_neighbors=no_neighbors,
                                                           use_all_neighbors_in_range=use_all_neighbors_in_range)
    else:
        prepared_data = select_kriging_data(unknown_position=unknown_location,
                                            data_array=known_locations,
                                            neighbors_range=neighbors_range,
                                            number_of_neighbors=no_neighbors,
                                            use_all_neighbors_in_range=use_all_neighbors_in_range)

    unknown_distances = prepared_data[:, -1]

    n = len(prepared_data)
    k = theoretical_model.predict(unknown_distances)
    k = k.T

    dists = calc_point_to_point_distance(prepared_data[:, :2])

    predicted_weights = theoretical_model.predict(dists.ravel())
    predicted = np.array(predicted_weights.reshape(n, n))
    return [k, predicted, prepared_data]


def solve_weights(weights: np.ndarray, k: np.ndarray, allow_lsa=False) -> np.ndarray:
    """
    Function solves Kriging System.

    Parameters
    ----------
    weights : numpy array
              Array of weights of size m:m.

    k : numpy array
        Array of semivariances of size m:n.

    allow_lsa : bool, default = False
                Allow algorithm to use Least Squares Approximation when solver fails.

    Returns
    -------
    solved : numpy array
             Final weights to estimate predicted value.

    Warns
    -----
    ZerosMatrixWarning : raised when weights / k matrices are full of zeros.

    TODO
    ----
    - functional tests and analysis of LSA output
    """

    try:
        solved = np.linalg.solve(weights, k)
    except np.linalg.LinAlgError as linalgerr:
        if np.mean(weights) == 0 or np.mean(k) == 0:
            warnings.warn(ZerosMatrixWarning().__str__())
            solved = np.zeros(len(k))
        else:
            if allow_lsa:
                warnings.warn(LeastSquaresApproximationWarning().__str__())
                solved = np.linalg.lstsq(weights, k)
                solved = solved[0]
            else:
                raise linalgerr

    return solved
