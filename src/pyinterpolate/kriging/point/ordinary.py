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
from geopandas import GeoSeries
from geopandas.array import GeometryArray
from numpy.typing import ArrayLike

from shapely.geometry import Point
from tqdm import tqdm

from pyinterpolate.core.data_models.points import VariogramPoints, \
    InterpolationPoints
from pyinterpolate.kriging.utils.errors import singular_matrix_error
# Pyinterpolate
from pyinterpolate.kriging.utils.point_kriging_solve import (get_predictions,
                                                             solve_weights)
from pyinterpolate.transform.statistical import sem_to_cov
from pyinterpolate.semivariogram.theoretical.theoretical import TheoreticalVariogram


def ok_calc(
        theoretical_model: TheoreticalVariogram,
        known_locations: np.ndarray,
        unknown_location: ArrayLike,
        neighbors_range=None,
        no_neighbors=4,
        max_tick=5.,
        use_all_neighbors_in_range=False,
        allow_approximate_solutions=False
):
    """
    Function predicts value at unknown location with Ordinary Kriging
    technique.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        Fitted theoretical variogram model.

    known_locations : numpy array
        Known locations: ``[x, y, value]``.

    unknown_location : Union[ArrayLike, Point]
        Points where you want to estimate value ``(x, y) <-> (lon, lat)``.

    neighbors_range : float, default=None
        The maximum distance where we search for neighbors. If ``None`` is
        given then the range is selected from the Theoretical
        Model's ``rang`` attribute.

    no_neighbors : int, default = 4
        The number of **n-closest neighbors** used for interpolation.

    max_tick : float, default=5.
        If searching for neighbors in a specific direction how big should be
        a tolerance for increasing the search angle (how many degrees more).

    use_all_neighbors_in_range : bool, default = False
        ``True``: if the real number of neighbors within
        the ``neighbors_range`` is greater than the ``number_of_neighbors``
        then take all of them anyway.

    allow_approximate_solutions : bool, default=False
        Allows the approximation of kriging weights based on the OLS
        algorithm. We don't recommend set it to ``True`` if you don't know
        what are you doing. This parameter can be useful when you have
        clusters in your dataset, that can lead to singular or near-singular
        matrix creation.

    Returns
    -------
    : numpy array
        ``[predicted value, variance error, longitude (x), latitude (y)]``

    Raises
    ------
    RunetimeError
        Singular Matrix in the Kriging system.
    """
    # Check if known locations are in the right format
    known_locations = VariogramPoints(known_locations).points

    # Check if unknown location is Point
    if isinstance(unknown_location, Point):
        unknown_location = (
            unknown_location.x,
            unknown_location.y
        )
        unknown_location = np.array(unknown_location)

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
        output_weights = solve_weights(weights,
                                       k,
                                       allow_approximate_solutions)
        zhat = dataset[:, -2].dot(output_weights[:-1])

        sigma = np.matmul(output_weights.T, k)

        if sigma < 0:
            return [zhat, np.nan, unknown_location[0], unknown_location[1]]

        return [zhat, sigma, unknown_location[0], unknown_location[1]]

    except np.linalg.LinAlgError as _:
        singular_matrix_error()


def ordinary_kriging(
        theoretical_model: TheoreticalVariogram,
        known_locations: ArrayLike,
        unknown_locations: Union[np.ndarray, Point, List, Tuple, GeoSeries, GeometryArray, ArrayLike],
        neighbors_range=None,
        no_neighbors=4,
        max_tick=5.,
        use_all_neighbors_in_range=False,
        allow_approximate_solutions=False,
        progress_bar: bool = True
) -> np.ndarray:
    """
    Function predicts value at unknown location with Ordinary Kriging
    technique.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        Fitted theoretical variogram model.

    known_locations : numpy array
        Known locations: ``[x, y, value]``.

    unknown_locations : Union[ArrayLike, Point]
        Points where you want to estimate value ``(x, y) <-> (lon, lat)``.

    neighbors_range : float, default=None
        The maximum distance where we search for neighbors. If ``None`` is
        given then the range is selected from the Theoretical
        Model's ``rang`` attribute.

    no_neighbors : int, default = 4
        The number of **n-closest neighbors** used for interpolation.

    max_tick : float, default=5.
        If searching for neighbors in a specific direction how big should be
        a tolerance for increasing the search angle (how many degrees more).

    use_all_neighbors_in_range : bool, default = False
        ``True``: if the real number of neighbors within
        the ``neighbors_range`` is greater than the ``number_of_neighbors``
        then take all of them anyway.

    allow_approximate_solutions : bool, default=False
        Allows the approximation of kriging weights based on the OLS
        algorithm. We don't recommend set it to ``True`` if you don't know
        what are you doing. This parameter can be useful when you have
        clusters in your dataset, that can lead to singular or near-singular
        matrix creation.

    progress_bar : bool, default=True
        Show a progress bar during the interpolation process.

    Returns
    -------
    : numpy array
        ``[predicted value, variance error, longitude (x), latitude (y)]``

    Raises
    ------
    RunetimeError
        Singular Matrix in the Kriging system.
    """
    # Check if known locations are in the right format
    known_locations = VariogramPoints(known_locations).points
    unknown_locations = InterpolationPoints(unknown_locations).points

    interpolated_results = []

    _disable_progress_bar = not progress_bar

    for upoints in tqdm(unknown_locations, disable=_disable_progress_bar):
        res = ok_calc(
            theoretical_model=theoretical_model,
            known_locations=known_locations,
            unknown_location=upoints,
            neighbors_range=neighbors_range,
            no_neighbors=no_neighbors,
            max_tick=max_tick,
            use_all_neighbors_in_range=use_all_neighbors_in_range,
            allow_approximate_solutions=allow_approximate_solutions
        )

        interpolated_results.append(
            res
        )

    return np.array(interpolated_results)


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
    Function predicts value at unknown location using Ordinary Kriging.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        A trained theoretical variogram model.

    known_locations : numpy array
        The known locations: ``[x, y, value]``.

    unknown_location : Union[List, Tuple, numpy array]
        Point where you want to estimate value ``(x, y) <-> (lon, lat)``.

    sill : float
        The sill (``c(0)``) of a dataset.

    neighbors_range : float, default=None
        The maximum distance where we search for neighbors. If ``None`` is
        given then range is selected from the Theoretical Model's
        ``rang`` attribute.

    no_neighbors : int, default = 4
        The number of the **n-closest neighbors** used for interpolation.

    max_tick : float, default=5.
        If searching for neighbors in a specific direction how large should be
        tolerance for increasing the search angle (how many degrees more).

    use_all_neighbors_in_range : bool, default = False
        ``True``: if the real number of neighbors within
         the ``neighbors_range`` is greater than the
        ``number_of_neighbors`` parameter then take all of them anyway.

    allow_approximate_solutions : bool, default=False
        Allows the approximation of kriging weights based on the OLS
        algorithm. We don't recommend set it to ``True``
        if you don't know what are you doing. This parameter can be useful
        when you have clusters in your dataset,
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
        output_weights = solve_weights(weights,
                                       k,
                                       allow_approximate_solutions)
        zhat = dataset[:, -2].dot(output_weights[:-1])

        sigma = sill - np.matmul(output_weights.T, k)

        if sigma < 0:
            return [zhat, np.nan, unknown_location[0], unknown_location[1]]

        return [zhat, sigma, unknown_location[0], unknown_location[1]]
    except np.linalg.LinAlgError as _:
        singular_matrix_error()
