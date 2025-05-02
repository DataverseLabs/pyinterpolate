"""
Distance calculation functions.

Authors
-------
1. Szymon Moliński | @SimonMolinsky

TODO
- eval with IDW
"""
from typing import Iterable

import numpy as np

from pyinterpolate.core.data_models.points import VariogramPoints
from pyinterpolate.distance.point import point_distance


def inverse_distance_weighting(known_locations: np.ndarray,
                               unknown_location: Iterable,
                               no_neighbors=-1,
                               power=2.) -> float:
    """
    Inverse Distance Weighting with a given set of points and
    the unknown location.

    Parameters
    ----------
    known_locations : numpy array
        The MxN array, where **M** is a number of rows (points) and **N**
        is the number of columns, where the last column represents a value
        observed in a known point. (It could be **(N-1)**-dimensional data).

    unknown_location : Iterable
        Array or list with coordinates of the unknown point.
        Its length is N-1 (number of dimensions). The unknown
        location `shape` should be the same as the ``known_points``
        parameter `shape`, if not, then new dimension
        is added once - vector of points ``[x, y]``
        becomes ``[[x, y]]`` for 2-dimensional data.

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
    """

    # Check power parameter
    if power < 0:
        raise ValueError('Power cannot be smaller than 0')

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
