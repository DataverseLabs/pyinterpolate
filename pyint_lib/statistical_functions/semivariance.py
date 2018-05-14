import numpy as np
from helper_functions.euclidean_distance import calculate_distance


def calculate_semivariance(points_array, lags, step_size):
    """
    Function calculates semivariance of points in n-dimensional space.
    :param points_array: numpy array of points and values (especially DEM) where
    points_array[0] = array([point_x, point_y, ..., point_n, value])
    :param lags: array of lags between points
    :param step_size: distance which should be included in the gamma parameter which enhances range of interest
    :return: semivariance: numpy array of pair of lag and semivariance values where
    semivariance[0] = array of lags
    semivariance[1] = array of lag's values

    """
    smv = []
    semivariance = []

    # Calculate distance
    distance_array = calculate_distance(points_array[:, 0:-1])

    # Calculate semivariances
    for h in lags:
        gammas = []
        distances_in_range = np.where(
            np.logical_and(
                np.greater_equal(distance_array, h - step_size), np.less_equal(distance_array, h + step_size)))
        for i in range(0, len(distances_in_range[0])):
            row_x = distances_in_range[0][i]
            row_x_h = distances_in_range[1][i]
            g = (points_array[row_x][2] - points_array[row_x_h][2]) ** 2
            gammas.append(g)
        if len(gammas) == 0:
            gamma = 0
        else:
            gamma = np.sum(gammas) / (2 * len(gammas))
        smv.append(gamma)

    # Selecting semivariances
    for i in range(len(lags)):
        if smv[i] > 0:
            semivariance.append([lags[i], smv[i]])

    semivariance = np.vstack(semivariance)

    return semivariance.T
