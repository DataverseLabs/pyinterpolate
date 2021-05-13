import numpy as np
from pyinterpolate.distance.calculate_distances import calc_point_to_point_distance


def inverse_distance_weighting(known_points, unknown_location, number_of_neighbours=-1, power=2.):
    """
    Function performs Inverse Distance Weighting with a given set of points and an unknown location.

    INPUT:

    :param known_points: (numpy array) [x, y, value],
    :param unknown_location: (numpy array) [[ux, uy]],
    :param number_of_neighbours: (int) default=-1 which means that all known points will be used to estimate value at
        the unknown location. Can be any number within the limits [2, length(known_points)],
    :param power: (float) default=2, power value must be larger or equal to 0, controls weight assigned to each known
        point.

    OUTPUT:

    :return: (float) value at unknown location.
    """

    # Test unknown points
    try:
        _ = unknown_location.shape[1]
    except IndexError:
        unknown_location = np.expand_dims(unknown_location, axis=0)

    # Test power

    if power < 0:
        raise ValueError('Power cannot be smaller than 0')

    # Test number of neighbours

    if number_of_neighbours == -1:
        number_of_closest = len(known_points)
    elif (number_of_neighbours >= 2) and (number_of_neighbours <= len(known_points)):
        number_of_closest = number_of_neighbours
    else:
        raise ValueError(f'Number of closest neighbors must be between 2 and the number of known points '
                         f'({len(known_points)}) and {number_of_neighbours} neighbours were given instead.')

    # Calculate distances

    distances = calc_point_to_point_distance(unknown_location, known_points[:, :-1])

    # Check if any distance is equal to 0
    if not np.all(distances[0]):

        zer_pos = np.where(distances == 0)
        unkn_value = known_points[zer_pos[1], -1][0]
        return unkn_value

    # Get n closest neighbours...

    sdists = distances.argsort()
    sdists = sdists[0, :number_of_closest]
    dists = distances[0, sdists]
    values = known_points[sdists].copy()
    values = values[:, -1]
    weights = 1 / dists**power

    unkn_value = np.sum(weights * values) / np.sum(weights)
    return unkn_value
