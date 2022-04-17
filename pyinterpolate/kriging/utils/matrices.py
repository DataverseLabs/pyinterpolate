import warnings
from typing import List, Union, Tuple

import numpy as np

from pyinterpolate.distance.distance import calc_point_to_point_distance
from pyinterpolate.kriging.utils.exceptions import ZerosMatrixWarning, LeastSquaresApproximationWarning
from pyinterpolate.processing.select_values import select_kriging_data
from pyinterpolate.variogram import TheoreticalVariogram
from pyinterpolate.variogram.utils.exceptions import validate_theoretical_variogram


def get_predictions(theoretical_model: TheoreticalVariogram,
                    known_locations: np.ndarray,
                    unknown_location: Union[List, Tuple, np.ndarray],
                    neighbors_range=None,
                    min_no_neighbors=1,
                    max_no_neighbors=-1) -> List:
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

    min_no_neighbors : int, default = 1
                       Minimum number of neighbors to estimate unknown value; value is used when insufficient number of
                       neighbors is within neighbors_range.

    max_no_neighbors : int, default = -1
                       Maximum number of n-closest neighbors used for interpolation if there are too many neighbors
                       in neighbors_range. It speeds up calculations for large datasets. Default -1 means that
                       all possible neighbors will be used.

    Returns
    -------
    : List[predictions - unknown point, predictions - point to point, prepared Kriging data]

    Raises
    ------
    VariogramModelNotSetError : Semivariogram model has not been set (it doesn't have a name)
    """
    # Check if variogram model is valid
    validate_theoretical_variogram(theoretical_model)

    # Check range
    if neighbors_range is None:
        neighbors_range = theoretical_model.rang

    prepared_data = select_kriging_data(unknown_position=unknown_location,
                                        data_array=known_locations,
                                        neighbors_range=neighbors_range,
                                        min_number_of_neighbors=min_no_neighbors,
                                        max_number_of_neighbors=max_no_neighbors)

    n = len(prepared_data)
    unknown_distances = prepared_data[:, -1]
    k = theoretical_model.predict(unknown_distances)
    k = k.T

    dists = calc_point_to_point_distance(prepared_data[:, :-2])

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
    """

    try:
        solved = np.linalg.solve(weights, k)
    except np.linalg.LinAlgError as linalgerr:
        if allow_lsa:
            if np.mean(weights) == 0 or np.mean(k) == 0:
                warnings.warn(ZerosMatrixWarning().__str__())
                solved = np.zeros(len(k))
            else:
                warnings.warn(LeastSquaresApproximationWarning().__str__())
                solved = np.linalg.lstsq(weights, k)
                solved = solved[0]
        else:
            raise linalgerr

    return solved
