from pyinterpolate.distance.calculate_distances import calc_point_to_point_distance


def inverse_distance_weighting(known_points, unknown_locations, number_of_neighbours=-1, power=2.):
    """
    Function perform Inverse Distance Weighting on a given set of points.

    INPUT:

    :param known_points: (numpy array) [x, y, value],
    :param unknown_locations: (numpy array) [[ux, uy]],
    :param number_of_neighbours: (int) default=-1 which means that all known points will be used to estimate value at
        the unknown location. Can be any number in the limits [2, length(known_points)],
    :param power: (float) controls weight assigned to each known point,
    """

    # Calculate distances

    distances = calc_point_to_point_distance(unknown_locations, known_points)

    # For each unknown location get n closest neighbours...

    if number_of_neighbours == -1:
        number_of_closest = len(known_points)
    elif (number_of_neighbours >= 2) and (number_of_neighbours <= len(known_points)):
        number_of_closest = number_of_neighbours
    else:
        raise ValueError(f'Number of closest neighbors must be between 2 and the number of known points '
                         f'({len(known_points)}) and {number_of_neighbours} neighbours were given instead.')

    for dists in distances:
        d = 0