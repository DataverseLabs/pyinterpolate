import numpy as np


def semivariance_fn(v1, v2):
    """
    Calculates semivariance.

    Parameters
    ----------
    v1 : numpy array
        1-Dimensional array with the points values of length N.

    v2 : numpy array
        1-Dimensional array with the neighbors of points from ``coor1`` of
        length N.

    Returns
    -------
    : float
        Semivariance value.
    """
    sem = np.square(v1 - v2)
    sem_value = np.mean(sem) / 2
    return sem_value
