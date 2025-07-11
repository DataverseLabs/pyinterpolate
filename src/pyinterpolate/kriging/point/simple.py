"""
Perform point simple kriging.

Authors
-------
1. Szymon Moliński | @SimonMolinsky

"""
from typing import List, Union, Tuple

import numpy as np
from geopandas import GeoSeries
from geopandas.array import GeometryArray
from numpy.typing import ArrayLike
from shapely.geometry import Point
from tqdm import tqdm

from pyinterpolate.core.data_models.points import VariogramPoints, \
    InterpolationPoints
from pyinterpolate.kriging.utils.errors import singular_matrix_error
from pyinterpolate.kriging.utils.point_kriging_solve import (get_predictions,
                                                             solve_weights)
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram


def sk_calc(
        theoretical_model: TheoreticalVariogram,
        known_locations: np.ndarray,
        unknown_location: Union[List, Tuple, np.ndarray],
        process_mean: float,
        neighbors_range=None,
        no_neighbors=4,
        max_tick=5.,
        use_all_neighbors_in_range=False,
        allow_approximate_solutions=False
) -> List:
    """
    Function predicts value at unknown location using Simple Kriging.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        Fitted theoretical variogram model.

    known_locations : numpy array
        The known locations: ``[x, y, value]``.

    unknown_location : Union[List, Tuple, numpy array]
        Point where you want to estimate value ``(x, y) <-> (lon, lat)``.

    process_mean : float
        The mean value of a process over a study area. Should be known
        before processing. That's why Simple Kriging has a limited number
        of applications. You must have multiple samples and well-known area
        to know this parameter.

    neighbors_range : float, default=None
        The maximum distance where we search for neighbors.  If ``None`` is
        given then range is selected from the Theoretical Model.

    no_neighbors : int, default = 4
        The number of the **n-closest neighbors** used for interpolation.

    max_tick : float, default=5.
        If searching for neighbors in a specific direction how large should
        be tolerance for increasing the search angle (how many degrees more).

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
        Singular matrix in Kriging system.
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

    try:
        output_weights = solve_weights(predicted,
                                       k,
                                       allow_approximate_solutions)
        r = dataset[:, -2] - process_mean
        zhat = r.dot(output_weights)
        zhat = zhat + process_mean

        sigma = np.matmul(output_weights.T, k)

        if sigma < 0:
            return [zhat, np.nan, unknown_location[0], unknown_location[1]]

        return [zhat, sigma, unknown_location[0], unknown_location[1]]

    except np.linalg.LinAlgError as _:
        singular_matrix_error()


def simple_kriging(
        theoretical_model: TheoreticalVariogram,
        known_locations: np.ndarray,
        unknown_locations: Union[np.ndarray, Point, List, Tuple, GeoSeries, GeometryArray, ArrayLike],
        process_mean: float,
        neighbors_range=None,
        no_neighbors=4,
        max_tick=5.,
        use_all_neighbors_in_range=False,
        allow_approximate_solutions=False,
        progress_bar: bool = True
) -> List:
    """
    Function predicts value at unknown location using Simple Kriging.

    Parameters
    ----------
    theoretical_model : TheoreticalVariogram
        Fitted theoretical variogram model.

    known_locations : numpy array
        The known locations: ``[x, y, value]``.

    unknown_locations : Union[np.ndarray, Point, List, Tuple, GeoSeries, GeometryArray, ArrayLike]
        Point where you want to estimate value ``(x, y) <-> (lon, lat)``.

    process_mean : float
        The mean value of a process over a study area. Should be known
        before processing. That's why Simple Kriging has a limited number
        of applications. You must have multiple samples and well-known area
        to know this parameter.

    neighbors_range : float, default=None
        The maximum distance where we search for neighbors.  If ``None`` is
        given then range is selected from the Theoretical Model.

    no_neighbors : int, default = 4
        The number of the **n-closest neighbors** used for interpolation.

    max_tick : float, default=5.
        If searching for neighbors in a specific direction how large should
        be tolerance for increasing the search angle (how many degrees more).

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

    progress_bar : bool, default=True
        Show a progress bar during the interpolation process.

    Returns
    -------
    : numpy array
        ``[predicted value, variance error, longitude (x), latitude (y)]``

    Raises
    ------
    RunetimeError
        Singular matrix in Kriging system.
    """

    # Check if known locations are in the right format
    known_locations = VariogramPoints(known_locations).points
    unknown_locations = InterpolationPoints(unknown_locations).points

    interpolated_results = []

    _disable_progress_bar = not progress_bar

    for upoints in tqdm(unknown_locations, disable=_disable_progress_bar):
        res = sk_calc(
            theoretical_model=theoretical_model,
            known_locations=known_locations,
            unknown_location=upoints,
            process_mean=process_mean,
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
