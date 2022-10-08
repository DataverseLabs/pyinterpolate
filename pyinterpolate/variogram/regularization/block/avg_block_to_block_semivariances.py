"""
Functions for calculating the average block-to-block semivariances

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
import numpy as np

from pyinterpolate.processing.select_values import select_values_in_range


def average_block_to_block_semivariances(semivariances_array: np.ndarray,
                                         lags: np.ndarray,
                                         step_size: float) -> np.ndarray:
    """
    Function averages block to block semivariances over specified lags.

    Parameters
    ----------
    semivariances_array : numpy array
                          [lag, semivariance, number of point pairs between blocks]

    lags : numpy array
           Array of lags.

    step_size : float

    Returns
    -------
    averaged : numpy array
               [lag, mean semivariances in a range, number of point pairs in range]
    """

    averaged = []
    distances = semivariances_array[:, 0]
    for lag in lags:
        distances_in_range = select_values_in_range(distances, lag, step_size)
        ldist = len(distances_in_range[0])
        if ldist > 0:
            semivars_in_range = semivariances_array[distances_in_range[0], 1]
            averaged.append([
                lag,
                np.mean(semivars_in_range),
                ldist
            ])
        else:
            averaged.append([
                lag,
                0,
                0
            ])
    averaged = np.array(averaged)
    return averaged
