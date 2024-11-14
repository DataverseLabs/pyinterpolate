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
from numpy.typing import ArrayLike

# Pyinterpolate
from pyinterpolate.kriging.utils.point_kriging_solve import get_predictions, solve_weights
from pyinterpolate.transform.statistical import sem_to_cov
from pyinterpolate.semivariogram.theoretical.theoretical import TheoreticalVariogram


def ordinary_kriging(
        theoretical_model: TheoreticalVariogram,
        known_locations: np.ndarray,
        unknown_location: ArrayLike,
        neighbors_range=None,
        no_neighbors=4,
        max_tick=5.,
        use_all_neighbors_in_range=False,
        allow_approximate_solutions=False
) -> np.ndarray:
    """
    Function predicts value at unknown location with Ordinary Kriging technique.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        Fitted theoretical variogram model.

    known_locations : numpy array
        The known locations.

    unknown_location : ArrayLike
        Points where you want to estimate value ``(x, y) <-> (lon, lat)``.

    neighbors_range : float, default=None
        The maximum distance where we search for neighbors. If ``None`` is given then range is selected from
        the ``theoretical_model`` ``rang`` attribute.

    no_neighbors : int, default = 4
        The number of the **n-closest neighbors** used for interpolation.

    max_tick : float, default=5.
        If searching for neighbors in a specific direction how big should be a tolerance for increasing
        the search angle (how many degrees more).

    use_all_neighbors_in_range : bool, default = False
        ``True``: if the real number of neighbors within the ``neighbors_range`` is greater than the
        ``number_of_neighbors`` parameter then take all of them anyway.

    allow_approximate_solutions : bool, default=False
        Allows the approximation of kriging custom_weights based on the OLS algorithm. We don't recommend set it to ``True``
        if you don't know what are you doing. This parameter can be useful when you have clusters in your dataset,
        that can lead to singular or near-singular matrix creation.

    Returns
    -------
    : numpy array
        ``[predicted value, variance error, longitude (x), latitude (y)]``

    Raises
    ------
    RunetimeError
        Singularity matrix in the Kriging system.
    """

    k, predicted, dataset = get_predictions(theoretical_model,
                                            known_locations,
                                            unknown_location,
                                            neighbors_range,
                                            no_neighbors,
                                            use_all_neighbors_in_range,
                                            max_tick)

    k_ones = np.ones(1)[0]
    k = np.r_[k, k_ones]

    p_ones = np.ones((predicted.shape[0], 1))
    predicted_with_ones_col = np.c_[predicted, p_ones]
    p_ones_row = np.ones((1, predicted_with_ones_col.shape[1]))
    p_ones_row[0][-1] = 0.
    weights = np.r_[predicted_with_ones_col, p_ones_row]

    try:
        output_weights = solve_weights(weights, k, allow_approximate_solutions)
    except np.linalg.LinAlgError as _:
        msg = "Singular matrix in Kriging system detected, check if you have duplicated coordinates " \
              "in the ``known_locations`` variable. If your data doesn't have duplicates then set " \
              "``allow_lsa`` parameter to ``True``."
        raise RuntimeError(msg)

    zhat = dataset[:, -2].dot(output_weights[:-1])

    sigma = np.matmul(output_weights.T, k)

    if sigma < 0:
        return [zhat, np.nan, unknown_location[0], unknown_location[1]]

    return [zhat, sigma, unknown_location[0], unknown_location[1]]


def ordinary_kriging_from_cov(
        theoretical_model: TheoreticalVariogram,
        known_locations: np.ndarray,
        unknown_location: Union[List, Tuple, np.ndarray],
        sill=None,
        neighbors_range=None,
        no_neighbors=4,
        max_tick=5.,
        use_all_neighbors_in_range=False,
        allow_approximate_solutions=False
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

    sill : float
        The sill (``c(0)``) of a dataset.

    neighbors_range : float, default=None
        The maximum distance where we search for neighbors. If ``None`` is given then range is selected from
        the ``theoretical_model`` ``rang`` attribute.

    no_neighbors : int, default = 4
        The number of the **n-closest neighbors** used for interpolation.

    max_tick : float, default=5.
        If searching for neighbors in a specific direction how big should be a tolerance for increasing
        the search angle (how many degrees more).

    use_all_neighbors_in_range : bool, default = False
        ``True``: if the real number of neighbors within the ``neighbors_range`` is greater than the
        ``number_of_neighbors`` parameter then take all of them anyway.

    allow_approximate_solutions : bool, default=False
        Allows the approximation of kriging custom_weights based on the OLS algorithm. We don't recommend set it to ``True``
        if you don't know what are you doing. This parameter can be useful when you have clusters in your dataset,
        that can lead to singular or near-singular matrix creation.

    Returns
    -------
    : numpy array
        ``[predicted value, variance error, longitude (x), latitude (y)]``

    Raises
    ------
    RunetimeError
        Singularity matrix in a Kriging system.
    """

    k, predicted, dataset = get_predictions(theoretical_model,
                                            known_locations,
                                            unknown_location,
                                            neighbors_range,
                                            no_neighbors,
                                            use_all_neighbors_in_range,
                                            max_tick)

    if sill is None:
        sill = theoretical_model.sill

    k = sem_to_cov(k, sill)
    predicted = sem_to_cov(predicted, sill)

    k_ones = np.ones(1)[0]
    k = np.r_[k, k_ones]

    p_ones = np.ones((predicted.shape[0], 1))
    predicted_with_ones_col = np.c_[predicted, p_ones]
    p_ones_row = np.ones((1, predicted_with_ones_col.shape[1]))
    p_ones_row[0][-1] = 0.
    weights = np.r_[predicted_with_ones_col, p_ones_row]

    try:
        output_weights = solve_weights(weights, k, allow_approximate_solutions)
    except np.linalg.LinAlgError as _:
        msg = 'Singular matrix in Kriging system detected, check if you have duplicated coordinates ' \
              'in the ``known_locations`` variable.'
        raise RuntimeError(msg)

    zhat = dataset[:, -2].dot(output_weights[:-1])

    sigma = sill - np.matmul(output_weights.T, k)

    if sigma < 0:
        return [zhat, np.nan, unknown_location[0], unknown_location[1]]

    return [zhat, sigma, unknown_location[0], unknown_location[1]]
