"""
Gets and solves Kriging System weights.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
import warnings
from typing import List

import numpy as np
from numpy.typing import ArrayLike

from pyinterpolate.distance.point import point_distance
from pyinterpolate.core.ptp_warnings.ptp_warnings import ZerosMatrixWarning, LeastSquaresApproximationWarning
from pyinterpolate.transform.select_points import select_kriging_data, select_kriging_data_from_direction
from pyinterpolate.semivariogram.theoretical.theoretical import TheoreticalVariogram


def get_predictions(theoretical_model: TheoreticalVariogram,
                    known_locations: np.ndarray,
                    unknown_location: ArrayLike,
                    neighbors_range=None,
                    no_neighbors=4,
                    use_all_neighbors_in_range=False,
                    max_tick=5) -> List:
    """
    Function predicts semivariances for distances between known and unknown
    points pairs.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        Trained theoretical variogram model.

    known_locations : numpy array
        Locations with measured values: ``[x, y, value]``.

    unknown_location : Union[List, Tuple, numpy array]
        Point where you want to estimate value ``(x, y) <-> (lon, lat)``

    neighbors_range : float, default = None
        The maximum distance where we search for the ``unknown_location``
        neighbors. If ``None`` given then range is selected from
        the Theoretical Model.

    no_neighbors : int, default = 4
        Number of the n-closest neighbors used for interpolation.

    use_all_neighbors_in_range : bool, default = False
        If ``True``: if number of neighbors within the ``neighbors_range``
        is greater than the ``number_of_neighbors`` then take all of them
        for modeling.

    max_tick : float, default=5.
        If searching for neighbors in a specific direction how large should
        be tolerance for increasing the search angle (how many degrees more).

    Returns
    -------
    : List
        Predictions from distance to unknown point,
        predictions from distance between known points,
        and prepared Kriging data.
    """

    # Check range
    if neighbors_range is None:
        neighbors_range = theoretical_model.rang

    if theoretical_model.direction is not None:
        prepared_data = select_kriging_data_from_direction(
            unknown_position=unknown_location,
            data_array=known_locations,
            neighbors_range=neighbors_range,
            direction=theoretical_model.direction,
            number_of_neighbors=no_neighbors,
            use_all_neighbors_in_range=use_all_neighbors_in_range,
            max_tick=max_tick
        )
    else:
        prepared_data = select_kriging_data(
            unknown_position=unknown_location,
            data_array=known_locations,
            neighbors_range=neighbors_range,
            number_of_neighbors=no_neighbors,
            use_all_neighbors_in_range=use_all_neighbors_in_range
        )

    unknown_distances = prepared_data[:, -1]

    n = len(prepared_data)
    k = theoretical_model.predict(unknown_distances)
    k = k.T

    dists = point_distance(prepared_data[:, :2], prepared_data[:, :2])

    predicted_weights = theoretical_model.predict(dists.ravel())
    predicted = np.array(predicted_weights.reshape(n, n))

    return [k, predicted, prepared_data]


def solve_weights(weights: np.ndarray,
                  k: np.ndarray,
                  allow_lsa=False) -> np.ndarray:
    """
    Solves Kriging System.

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
    ZerosMatrixWarning :
        Weights or k matrices are zero-matrices.

    LeastSquaresApproximationWarning :
        Weights are solved using the LSA algorithm
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
                solved = np.linalg.lstsq(weights, k, rcond=None)
                solved = solved[0]
            else:
                raise linalgerr

    return solved
