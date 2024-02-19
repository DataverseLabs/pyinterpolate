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
    Calculates the euclidean distance from one group of points to another
    group of points.

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
        Distances matrix. Row index = ``points`` point index, and column
        index = ``other`` point index.

    Notes
    -----
        The function creates array of size MxN, where M = number of ``points``
        and N = number of ``other``. Very big array with coordinates may cause
        a memory error.

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


def select_values_between_lags(data: np.ndarray,
                               current_lag: float,
                               previous_lag: float):
    """
    Function selects set of values which are greater than
    (lag - step_size size) and smaller or equal to (lag).

    Parameters
    ----------
    data : numpy array
           Distances between points.

    current_lag : float

    previous_lag : float

    Returns
    -------
    : numpy array
        Mask with distances within a specified radius.
    """

    # Check conditions
    condition_matrix = np.logical_and(
        np.greater(data, previous_lag),
        np.less_equal(data, current_lag))

    # Find positions
    position_matrix = np.where(condition_matrix)
    return position_matrix
