"""
Distance calculation functions.

Authors
-------
1. Szymon Moliński | @SimonMolinsky

TODO
- eval with IDW
"""
from numpy.typing import ArrayLike

import numpy as np

from pyinterpolate.core.data_models.points import VariogramPoints
from pyinterpolate.distance.point import point_distance
from pyinterpolate.transform.geo import geometry_and_values_array


def inverse_distance_weighting(unknown_location: ArrayLike,
                               known_locations: ArrayLike = None,
                               known_values: ArrayLike = None,
                               known_geometries: ArrayLike = None,
                               no_neighbors=-1,
                               power=2.) -> float:
    """
    Inverse Distance Weighting with a given set of points and
    the unknown location.

    Parameters
    ----------
    unknown_location : Iterable
        Array or list with coordinates of the unknown point.
        Its length is N-1 (number of dimensions). The unknown
        location `shape` should be the same as the ``known_points``
        parameter `shape`, if not, then new dimension
        is added once - vector of points ``[x, y]``
        becomes ``[[x, y]]`` for 2-dimensional data.

    known_locations : numpy array, optional
        The known locations: ``[x, y, value]``.

    known_values : ArrayLike, optional
        Observation in the i-th geometry (from ``known_geometries``). Optional
        parameter, if not given then ``known_locations`` must be provided.

    known_geometries : ArrayLike, optional
        Array or similar structure with geometries. It must have the same
        length as ``known_values``. Optional parameter, if not given then
        ``known_locations`` must be provided. Point type geometry.

    no_neighbors : int, default = -1
        If default value **(-1)** then all known points will be used to
        estimate value at the unknown location.
        Can be any number within the limits ``[2, len(known_points)]``,

    power : float, default = 2.
        Power value must be larger or equal to 0. It controls weight
        assigned to each known point. Larger power means
        stronger influence of the closest neighbors, but it decreases faster.

    Returns
    -------
    result : float
        The estimated value.

    Raises
    ------
    ValueError
        Power parameter set to be smaller than 0.

    ValueError
        Less than 2 neighbours or more than the number of ``known_points``
        neighbours are given in the ``number_of_neighbours`` parameter.

    Examples
    --------
    >>> unknown_pos = (10, 10)
    >>> locs = np.array([
    ...     [11, 1, 1],
    ...     [23, 2, 2],
    ...     [33, 3, 3],
    ...     [14, 44, 4],
    ...     [13, 10, 9],
    ...     [12, 55, 35],
    ...     [11, 9, 7]
    ... ])
    >>> pred = inverse_distance_weighting(
    ...     unknown_locations=unknown_pos,
    ...     known_values=locs[:, -1],
    ...     known_geometries=locs[:, :-1],
    ...     no_neighbors=2
    ... )
    >>> print(pred)
    7.286311587314138
    """

    # Check power parameter
    if power < 0:
        raise ValueError('Power cannot be smaller than 0')

    # Get known locations
    if known_locations is None:
        known_locations = geometry_and_values_array(
            geometry=known_geometries,
            values=known_values
        )

    # Check known points parameter
    # Check if known locations are in the right format
    known_locations = VariogramPoints(known_locations).points

    # Check number of neighbours parameter
    nn_neighbors_ge_2 = no_neighbors >= 2
    nn_neighbors_le_known_points = no_neighbors <= len(known_locations)
    n_closest_eq_nn = nn_neighbors_ge_2 and nn_neighbors_le_known_points

    number_of_closest = len(known_locations)
    
    if no_neighbors == -1:
        pass
    elif n_closest_eq_nn:
        number_of_closest = no_neighbors
    else:
        _idw_value_error_nn(length_known=len(known_locations),
                            nn=no_neighbors)

    # Pre-process unknown location parameter
    if not isinstance(unknown_location, np.ndarray):
        unknown_location = np.array(unknown_location)

    if len(unknown_location.shape) != len(known_locations.shape):
        unknown_location = unknown_location[np.newaxis, ...]

    # Calculate distances
    distances = point_distance(unknown_location, known_locations[:, :-1])
    distances: np.ndarray

    # Check if any distance is equal to 0 - then return this value
    if not np.all(distances[0]):

        zer_pos = np.where(distances == 0)
        result = known_locations[zer_pos[1], -1][0]
        return result

    # Get n closest neighbours...
    sdists = distances.argsort()
    sdists = sdists[0, :number_of_closest]
    dists = distances[0, sdists]
    values = known_locations[sdists].copy()
    values = values[:, -1]

    # Create weights
    weights = 1 / dists**power

    # Estimate value
    result = np.sum(weights * values) / np.sum(weights)
    return result


def _idw_value_error_nn(length_known: int, nn: int):
    """
    Helper function to raise ValueError when the number of closest neighbours
    is out of bounds.

    Parameters
    ----------
    length_known : int
        Number of known points.

    nn : int
        Number of neighbours defined by the user.

    Raises
    ------
    ValueError
        Less than 2 neighbours or more than the number of ``known_points``
        neighbours are given in the ``number_of_neighbours`` parameter.

    """
    raise ValueError(
        f'Number of closest neighbors must be between 2 '
        f'and the number of known points '
        f'({length_known}) and {nn} neighbours were given instead.')
