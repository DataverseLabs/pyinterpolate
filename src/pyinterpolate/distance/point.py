"""
Distance calculation functions.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist


# noinspection PyTypeChecker
def point_distance(points: ArrayLike,
                   other: ArrayLike,
                   metrics: str = 'euclidean') -> np.ndarray:
    """
    Calculates the Euclidean distance from set of points to
    other set of points.

    Parameters
    ----------
    points : array
        Spatial coordinates.

    other : array
        Other array with spatial coordinates.

    metrics : str, default = 'euclidean'
        Metrics used to calculate distance.
        See ``scipy.spatial.distance.cdist`` for more details.

    Returns
    -------
    distances : array
        Distances matrix. Row index = ``points`` point index,
        and column index = ``other`` point index.

    Notes
    -----
    The function creates array of size MxN, where M = number of
    ``points`` and N = number of ``other``. Large arrays may cause
    memory errors.

    Examples
    --------
    >>> points = [(0, 0), (0, 1), (0, 2)]
    >>> other = [(2, 2), (3, 3)]
    >>> distances = point_distance(points=points, other=other)
    >>> print(distances)
    [[2.82842712 4.24264069]
     [2.23606798 3.60555128]
     [2.         3.16227766]]
    """

    distances = cdist(points, other, metrics)

    return distances


def select_values_in_range(data: np.ndarray,
                           current_lag: float,
                           previous_lag: float):
    """
    Function selects distances between lags.

    Parameters
    ----------
    data : numpy array
        Distances between points.

    current_lag : float
        Actual maximum distance.

    previous_lag : float
        Previous maximum distance.

    Returns
    -------
    : numpy array
        Mask with distances between the previous maximum distance and
        the actual maximum distance.
    """

    # Check conditions
    condition_matrix = np.logical_and(
            np.greater(data, previous_lag),
            np.less_equal(data, current_lag))

    # Find positions
    position_matrix = np.where(condition_matrix)
    return position_matrix
