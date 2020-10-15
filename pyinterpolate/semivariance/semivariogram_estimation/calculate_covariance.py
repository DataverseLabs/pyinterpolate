import numpy as np

from pyinterpolate.calculations.distances.calculate_distances import calc_point_to_point_distance
from pyinterpolate.data_processing.data_preparation.select_values_in_range import select_values_in_range


def calculate_covariance(data, lags, step_size):
    """Function calculates covariance of a given set of points.

        Equation for calculation is:

            covariance = 1 / (N) * SUM(i=1, N) [z(x_i + h) * z(x_i)] - u^2

        where:

            - N - number of observation pairs,
            - h - distance (lag),
            - z(x_i) - value at location z_i,
            - (x_i + h) - location at a distance h from x_i,
            - u - mean observation at a given lag distance.

        INPUT:

        :param data: array of coordinates and their values,
        :param lags: array of lags between points,
        :param step_size: distance which should be included in the gamma parameter which enhances range of interest.

        OUTPUT:

        :return: covariance: numpy array of pair of lag and covariance values where:

            - covariance[0] = array of lags
            - covariance[1] = array of lag's values
            - covariance[2] = array of number of points in each lag.
    """

    distances = calc_point_to_point_distance(data[:, :-1])

    # Get only upper diagonal of distances, rest set to -1
    covar = []
    covariance = []

    for idx, h in enumerate(lags):
        distances_in_range = select_values_in_range(distances, h, step_size)
        cov = (data[distances_in_range[0], 2] * data[distances_in_range[1], 2])
        u_mean = (data[distances_in_range[0], 2] + data[distances_in_range[1], 2])
        u_mean = u_mean / (2 * len(u_mean))
        cov_value = np.sum(cov) / (len(cov)) - np.sum(u_mean) ** 2
        covar.append([cov_value, len(cov)])
        if covar[idx][0] > 0:
            covariance.append([h, covar[idx][0], covar[idx][1]])
        else:
            covariance.append([h, 0, 0])

    covariance = np.vstack(covariance)

    return covariance
