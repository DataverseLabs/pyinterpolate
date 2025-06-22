from typing import Union, List

import numpy as np

from pyinterpolate.distance.angular import select_points_within_ellipse, \
    define_whitening_matrix
from pyinterpolate.semivariogram.lags.lags import get_current_and_previous_lag


def from_ellipse_non_weighted(points: np.ndarray,
                              lags: Union[List, np.ndarray],
                              direction: float,
                              tolerance: float,
                              raise_when_no_neighbors: bool = False):
    """
    Function calculates semivariances from elliptical neighborhood.

    Parameters
    ----------
    points : numpy array
        Coordinates and their values: ``[x, y, value]``.

    lags : numpy array
        Bins array.

    direction : float
        Direction of semivariogram, values from 0 to 360 degrees:

        - 0 or 180: is E-W,
        - 90 or 270 is N-S,
        - 45 or 225 is NE-SW,
        - 135 or 315 is NW-SE.

    tolerance : float, default=1
        Value in range (0-1] to calculate semi-minor axis length of the
        search area. If tolerance is close to 0 then points must be placed
        in a single line with beginning in the origin of coordinate system
        and direction given by y axis and direction parameter.
            * The major axis length == step_size,
            * The minor axis size == tolerance * step_size.
            * Tolerance == 1 creates the omnidirectional semivariogram.

    # TODO: make it true default
    raise_when_no_neighbors : bool, default = True
        Raise ValueError if no neighbors in a bin.

    Returns
    -------
    output_semivariances : numpy array
        ``[lag, semivariance, number of point pairs]``
    """
    semivariances_and_lags = list()

    w_matrix = define_whitening_matrix(theta=direction,
                                       minor_axis_size=tolerance)

    coords = points[:, :-1]
    vector_distances = (
            coords[:, np.newaxis] - coords
    ).reshape(-1, coords.shape[1])

    values = points[:, -1]
    values = values[..., np.newaxis]

    point_diffs = (
            values[:, np.newaxis] - values
    ).flatten()

    for idx in range(len(lags)):
        semivars_per_lag = _get_semivars_not_weighted(
            coord_distances=vector_distances,
            value_differences=point_diffs,
            lags=lags,
            lag_idx=idx,
            w_matrix=w_matrix
        )

        if len(semivars_per_lag) == 0:
            if idx == 0 and lags[0] == 0:
                semivariances_and_lags.append([0, 0, 0])
            else:
                if raise_when_no_neighbors:
                    msg = f'There are no neighbors for a lag {lags[idx]},' \
                          f' the process has been stopped.'
                    raise RuntimeError(msg)
                else:
                    semivariances_and_lags.append([lags[idx], 0, 0])
        else:
            average_semivariance = np.mean(semivars_per_lag) / 2
            semivariances_and_lags.append(
                [lags[idx], average_semivariance, len(semivars_per_lag)])

    output_semivariances = np.array(semivariances_and_lags)
    return output_semivariances


def _get_semivars_not_weighted(coord_distances: np.ndarray,
                               value_differences: np.ndarray,
                               lags: Union[List, np.ndarray],
                               lag_idx: int,
                               w_matrix: np.ndarray) -> np.ndarray:
    """
    Function selects semivariances per lag in elliptical area.

    Parameters
    ----------
    coord_distances : numpy array
        Shape (N, 2)

    value_differences : numpy array
        Length N

    lags : list or numpy array
        The list of lags.

    lag_idx : int
        Current lag index.

    w_matrix : numpy array
        Matrix used for masking values in ellipse.

    Returns
    -------
    semivars_per_lag : numpy array
    """
    current_lag, previous_lag = get_current_and_previous_lag(
        lag_idx=lag_idx, lags=lags
    )

    step_size = current_lag - previous_lag

    masks = select_points_within_ellipse(
        coord_distances, current_lag, step_size, w_matrix
    )

    point_diffs_in_range = value_differences[masks]

    if len(point_diffs_in_range) > 0:
        semivars_per_lag = point_diffs_in_range ** 2
    else:
        semivars_per_lag = point_diffs_in_range

    return semivars_per_lag
