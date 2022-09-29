"""
Distance calculation functions.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Iterable

import numpy as np
from pyinterpolate.distance.distance import calc_point_to_point_distance


def inverse_distance_weighting(known_points: np.ndarray,
                               unknown_location: Iterable,
                               number_of_neighbours=-1,
                               power=2.) -> float:
    """Inverse Distance Weighting with a given set of points and an unknown location.

    Parameters
    ----------
    known_points : numpy array
        The MxN array, where **M** is a number of rows (points) and **N** is the number of columns, where the last
        column represents a value of a known point. (It could be **(N-1)**-dimensional data).

    unknown_location : Iterable
        Array or list with coordinates of the unknown point. It's length is N-1 (number of dimensions). The unknown
        location `shape` should be the same as the ``known_points`` parameter `shape`, if not, then new dimension
        is added once - vector of points ``[x, y]`` becomes ``[[x, y]]`` for 2-dimensional data.

    number_of_neighbours : int, default = -1
        If default value **(-1)** then all known points will be used to estimate value at the unknown location.
        Can be any number within the limits ``[2, len(known_points)]``,

    power : float, default = 2.
        Power value must be larger or equal to 0. It controls weight assigned to each known point. Larger power means
        stronger influence of the closest neighbors, but it decreases quickly.

    Returns
    -------
    result : float
        The estimated value.

    Raises
    ------
    ValueError
        Power parameter set to be smaller than 0.

    ValueError
        Less than 2 neighbours or more than the number of ``known_points`` neighbours are given in the
        ``number_of_neighbours`` parameter.
    """

    # Check power parameter
    if power < 0:
        raise ValueError('Power cannot be smaller than 0')

    # Check number of neighbours parameter
    if number_of_neighbours == -1:
        number_of_closest = len(known_points)
    elif (number_of_neighbours >= 2) and (number_of_neighbours <= len(known_points)):
        number_of_closest = number_of_neighbours
    else:
        raise ValueError(f'Number of closest neighbors must be between 2 and the number of known points '
                         f'({len(known_points)}) and {number_of_neighbours} neighbours were given instead.')

    # Pre-process unknown location parameter
    if not isinstance(unknown_location, np.ndarray):
        unknown_location = np.array(unknown_location)

    if len(unknown_location.shape) != len(known_points.shape):
        unknown_location = unknown_location[np.newaxis, ...]

    # Calculate distances
    distances = calc_point_to_point_distance(unknown_location, known_points[:, :-1])

    # Check if any distance is equal to 0 - then return this value
    if not np.all(distances[0]):

        zer_pos = np.where(distances == 0)
        result = known_points[zer_pos[1], -1][0]
        return result

    # Get n closest neighbours...
    sdists = distances.argsort()
    sdists = sdists[0, :number_of_closest]
    dists = distances[0, sdists]
    values = known_points[sdists].copy()
    values = values[:, -1]

    # Create weights
    weights = 1 / dists**power

    # Estimate value
    result = np.sum(weights * values) / np.sum(weights)
    return result
