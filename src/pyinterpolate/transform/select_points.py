from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike

from pyinterpolate.distance.angular import calc_angles, \
    calculate_angular_difference
from pyinterpolate.distance.point import point_distance


def create_min_max_array(value: float,
                         min_scaling_factor: float,
                         max_scaling_factor: float,
                         number_of_steps: int) -> np.ndarray:
    """
    Function prepares a numpy array of N equidistant values between (a:b),
    where:
      * N - number of steps,
      * a - min_scaling_factor * value,
      * b - max_scaling_factor * value.

    Parameters
    ----------
    value : float

    min_scaling_factor : float

    max_scaling_factor : float

    number_of_steps : int

    Returns
    -------
    : numpy array
    """
    min_step = value * min_scaling_factor
    max_step = value * max_scaling_factor
    min_max_steps = np.linspace(min_step, max_step, number_of_steps)
    return min_max_steps


def select_kriging_data(
        unknown_position: ArrayLike,
        data_array: np.ndarray,
        neighbors_range: float,
        number_of_neighbors: int = 4,
        use_all_neighbors_in_range: bool = False
) -> np.ndarray:
    """
    Function prepares data for kriging - array of point position,
    value and distance to an unknown point.

    Parameters
    ----------
    unknown_position : ArrayLike
        Single unknown location: ``(x, y)``

    data_array : numpy array
        Known points.

    neighbors_range : float
        Range within neighbors are affecting the value, it should be
        close or the same as the variogram range.

    number_of_neighbors : int, default = 4
        Number of the n-closest neighbors used for interpolation.

    use_all_neighbors_in_range : bool, default = False
        ``True``: if number of neighbors within the ``neighbors_range``
        is greater than the ``number_of_neighbors`` then use all
        neighbors in the range.

    Returns
    -------
    : numpy array
        Dataset of the length ``number_of_neighbors`` <= length. Each record
        is created from the position, value and distance to the unknown
        point ``[[x, y, value, distance to unknown position]]``.

    """

    # Distances to unknown point
    r = np.array([unknown_position])

    known_pos = data_array[:, :-1]
    dists = point_distance(r, known_pos)

    # Prepare data for kriging
    neighbors_and_dists = np.c_[data_array, dists.T]
    sorted_neighbors_and_dists = neighbors_and_dists[
        neighbors_and_dists[:, -1].argsort()]
    prepared_data = sorted_neighbors_and_dists[
                    sorted_neighbors_and_dists[:, -1] <= neighbors_range, :]

    if len(prepared_data) >= number_of_neighbors:
        if use_all_neighbors_in_range:
            return prepared_data
        else:
            return prepared_data[:number_of_neighbors]
    else:
        return sorted_neighbors_and_dists[:number_of_neighbors]


def select_kriging_data_from_direction(
        unknown_position: Iterable,
        data_array: np.ndarray,
        neighbors_range: float,
        direction: float,
        number_of_neighbors: int = 4,
        use_all_neighbors_in_range: bool = False,
        max_tick: int = 10
) -> np.ndarray:
    """
    Function selects closest neighbors based on the specific
    direction and tolerance.

    Parameters
    ----------
    unknown_position : Iterable
        List, tuple or array with x, y coordinates.

    data_array : numpy array
        Known points.

    neighbors_range : float
        Range within neighbors are affecting the value, it should be
        lower or the same as the variogram range.

    direction : float
        The direction of a directional variogram.

    number_of_neighbors : int, default = 4
        Number of the n-closest neighbors used for interpolation.

    use_all_neighbors_in_range : bool, default = False
        ``True``: if number of neighbors within the ``neighbors_range``
        is greater than the ``number_of_neighbors`` then
        take all of them for modeling.

    max_tick : int, default = 10
        How many degrees can be added to the search sphere.

    Returns
    -------
    : numpy array
        Dataset of the length number_of_neighbors <= length. Each record is
        created from the position, value and distance to the unknown
        point ``[[x, y, value, distance to unknown position]]``.
    """
    # Distances to unknown point
    r = np.array([unknown_position])

    known_pos = data_array[:, :-1]
    dists = point_distance(r, known_pos)
    angles = calc_angles(known_pos, origin=unknown_position)
    angle_diffs = calculate_angular_difference(angles, direction)

    selected_data = select_possible_neighbors_angular(
        data_array,
        dists,
        angle_diffs,
        neighbors_range,
        number_of_neighbors,
        use_all_neighbors_in_range,
        max_tick
    )

    return selected_data[:, :-1]


def select_possible_neighbors_angular(possible_neighbors: np.ndarray,
                                      distances: np.ndarray,
                                      angle_differences: np.ndarray,
                                      max_range: float,
                                      min_number_of_neighbors: int,
                                      use_all_neighbors_in_range: bool,
                                      max_tick) -> np.ndarray:
    """
    Function selects possible neighbors.

    Parameters
    ----------
    possible_neighbors : numpy array
        The array with possible neighbors.

    distances : numpy array
        The array with distances to each neighbor.

    angle_differences : numpy array
        The array with the minimal direction between expected distance and
        other points.

    max_range : float
        Range within neighbors are affecting the value, it should be lower
        or the same as the variogram range.

    min_number_of_neighbors : int
        Number of the n-closest neighbors used for interpolation.

    use_all_neighbors_in_range : bool
        ``True``: if number of neighbors within the ``neighbors_range``
        is greater than the ``number_of_neighbors`` then take all of
        them for modeling.

    max_tick : int
        How many degrees can be added to the search sphere.

    Returns
    -------
    possible_neighbors : numpy array
        Sorted neighbors based on a distance and direction from the origin.

    Raises
    ------
    ValueError : no neighbors in specified range
    """
    angular_tolerance = 1

    # Select those in distance range
    distances_in_range_mask = distances <= max_range
    distances_in_range_mask = distances_in_range_mask[0]

    possible_neighbors = possible_neighbors[distances_in_range_mask, :]
    distances = distances[:, distances_in_range_mask]
    angle_differences = angle_differences[distances_in_range_mask]

    # Sort based on a distance
    sorted_distances_in_range_mask = distances.argsort()
    sorted_distances_in_range_mask = sorted_distances_in_range_mask[0]

    possible_neighbors = possible_neighbors[sorted_distances_in_range_mask, :]
    distances = distances[:, sorted_distances_in_range_mask]
    angle_differences = angle_differences[sorted_distances_in_range_mask]

    prepared_data = np.c_[possible_neighbors, distances.T, angle_differences.T]

    # Limit to a specific direction
    prepared_data_with_angles = prepared_data[
        angle_differences <= angular_tolerance]

    while len(prepared_data_with_angles) < min_number_of_neighbors:
        angular_tolerance = angular_tolerance + 1
        prepared_data_with_angles = prepared_data[
            angle_differences <= angular_tolerance]
        if angular_tolerance > max_tick:
            break

    if len(prepared_data_with_angles) >= min_number_of_neighbors:
        if use_all_neighbors_in_range:
            return prepared_data_with_angles
        else:
            return prepared_data_with_angles[:min_number_of_neighbors]
    else:
        if len(prepared_data_with_angles) > 0:
            return prepared_data_with_angles
        else:
            return np.array([[np.nan, np.nan, np.nan, np.nan, np.nan]])


# if __name__ == '__main__':
#     import datetime
#
#     unknown_positions = np.random.random(size=(1000, 2))
#     known_positions = np.random.random(size=(5000, 3))
#
#     t0 = datetime.datetime.now()
#     for uk in unknown_positions:
#         _ = select_kriging_data_from_direction(
#             unknown_position=uk,
#             data_array=known_positions,
#             neighbors_range=0.1,
#             direction=15,
#             number_of_neighbors=4,
#             max_tick=5
#         )
#     tx = datetime.datetime.now() - t0
#     print(tx.total_seconds())
