import numpy as np
from pyinterpolate.kriging.helper_functions.euclidean_distance import calculate_distance


def calculate_covariance(points_array, lags, step_size):
    """
        Function calculates covariance of points in n-dimensional space.
        :param points_array: numpy array of points and values (especially DEM) where
        points_array[0] = array([point_x, point_y, ..., point_n, value])
        :param lags: array of lags between points
        :param step_size: distance which should be included in the gamma parameter which enhances range of interest
        :return: covariance: numpy array of pair of lag and covariance values where
        covariance[0] = array of lags
        covariance[1] = array of values for each lag

    """
    covariance = []

    # Calculate distance
    distance_array = calculate_distance(points_array[:, 0:-1])

    # Calculate covariance
    for h in lags:
        cov = []
        mu = 0
        distances_in_range = np.where(
            np.logical_and(
                np.greater_equal(distance_array, h - step_size), np.less_equal(distance_array, h + step_size)))
        for i in range(0, len(distances_in_range[0])):
            row_x = distances_in_range[0][i]
            row_x_h = distances_in_range[1][i]
            c = (points_array[row_x][2] * points_array[row_x_h][2])
            cov.append(c)
            mu += (points_array[row_x][2] + points_array[row_x_h][2])
        if len(cov) == 0:
            cvr = 0
        else:
            mu = mu / (2 * len(cov))
            cvr = np.sum(cov) / len(cov) - mu ** 2
        covariance.append(cvr)

    output_covariance = []
    for i in range(len(lags)):
        output_covariance.append([lags[i], covariance[i]])

    output_covariance = np.vstack(output_covariance)

    return output_covariance.T
