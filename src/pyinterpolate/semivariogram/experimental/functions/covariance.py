import numpy as np


def covariance_fn(points, neighbors):
    """
    Calculates covariance.

    Parameters
    ----------
    points : numpy array
        1-Dimensional array with the points values of length N.

    neighbors : numpy array
        1-Dimensional array with the neighbors of points from ``points`` of
        length N.

    Returns
    -------
    : float
        Covariance value.
    """
    lag_mean = np.mean(neighbors)  # it must be the mean for all neighbors...
    lag_mean_squared = lag_mean ** 2
    cov = (points * neighbors) - lag_mean_squared
    cov_value = np.mean(cov)
    return cov_value
