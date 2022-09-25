"""
Perform point ordinary kriging.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
# Python core
from typing import List, Union, Tuple

# Core calculation and data visualization
import numpy as np

# Pyinterpolate
from pyinterpolate.kriging.utils.process import get_predictions, solve_weights
from pyinterpolate.variogram import TheoreticalVariogram


def ordinary_kriging(
        theoretical_model: TheoreticalVariogram,
        known_locations: np.ndarray,
        unknown_location: Union[List, Tuple, np.ndarray],
        neighbors_range=None,
        no_neighbors=4,
        use_all_neighbors_in_range=False,
        allow_approximate_solutions=False,
        err_to_nan=False
) -> List:
    """
    Function predicts value at unknown location with Ordinary Kriging technique.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        A trained theoretical variogram model.

    known_locations : numpy array
        The known locations.

    unknown_location : Union[List, Tuple, numpy array]
        Point where you want to estimate value ``(x, y) <-> (lon, lat)``.

    neighbors_range : float, default=None
        The maximum distance where we search for neighbors. If ``None`` is given then range is selected from
        the ``theoretical_model`` ``rang`` attribute.

    no_neighbors : int, default = 4
        The number of the **n-closest neighbors** used for interpolation.

    use_all_neighbors_in_range : bool, default = False
        ``True``: if the real number of neighbors within the ``neighbors_range`` is greater than the
        ``number_of_neighbors`` parameter then take all of them anyway.

    allow_approx_solutions : bool, default=False
        Allows the approximation of kriging weights based on the OLS algorithm. We don't recommend set it to ``True``
        if you don't know what are you doing.

    err_to_nan : bool, default=False
        Return ``NaN`` if algorithm detects singular matrix.

    Returns
    -------
    : numpy array
        ``[predicted value, variance error, longitude (x), latitude (y)]``
    """

    k, predicted, dataset = get_predictions(theoretical_model,
                                            known_locations,
                                            unknown_location,
                                            neighbors_range,
                                            no_neighbors,
                                            use_all_neighbors_in_range)

    k_ones = np.ones(1)[0]
    k = np.r_[k, k_ones]

    p_ones = np.ones((predicted.shape[0], 1))
    predicted_with_ones_col = np.c_[predicted, p_ones]
    p_ones_row = np.ones((1, predicted_with_ones_col.shape[1]))
    p_ones_row[0][-1] = 0.
    weights = np.r_[predicted_with_ones_col, p_ones_row]

    if err_to_nan:
        try:
            output_weights = solve_weights(weights, k, allow_approximate_solutions)
        except np.linalg.LinAlgError as _:
            return [np.nan, np.nan, unknown_location[0], unknown_location[1]]
    else:
        output_weights = solve_weights(weights, k, allow_approximate_solutions)

    zhat = dataset[:, -2].dot(output_weights[:-1])

    sigma = np.matmul(output_weights.T, k)

    return [zhat, sigma, unknown_location[0], unknown_location[1]]
