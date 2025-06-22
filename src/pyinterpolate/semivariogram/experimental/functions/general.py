import concurrent.futures
from typing import Callable, Union, List
from collections import OrderedDict
from operator import itemgetter

import numpy as np

from pyinterpolate.distance.point import select_values_in_range, point_distance
from pyinterpolate.semivariogram.weights.experimental.weighting import \
    weight_experimental_semivariance


def omnidirectional_semivariogram_cloud(
        points: np.ndarray,
        lags: Union[List, np.ndarray],
        raise_when_no_neighbors: bool = False
):
    """
    Calculates omnidirectional semivariances cloud.

    Parameters
    ----------
    points : numpy array
        ``[x, y, value]``

    lags : list or numpy array
        The list of lags.

    raise_when_no_neighbors : bool, default = False
        Raise error when no neighbors are selected for a given lag.

    Returns
    -------
    : numpy array
        Ordered semivariances or covariances
        ``[lag, value, number of point pairs]``
    """

    coordinates = points[:, :-1]

    distances = point_distance(
        points=coordinates,
        other=coordinates
    )

    omnidirectional_values = []

    def _get(lag_idx):
        omnidirectional_values.append(
            _calc_omnidirectional_cloud(
                points=points,
                distances=distances,
                lags=lags,
                lag_idx=lag_idx,
                raise_when_no_neighbors=raise_when_no_neighbors
            )
        )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for idx in range(len(lags)):
            futures.append(
                executor.submit(
                    _get, idx
                )
            )
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                raise e

    # Clean cloud
    sorted_omnidirectional_values = _clean_cloud_data(omnidirectional_values)
    return sorted_omnidirectional_values


def omnidirectional_variogram(
        fn: Callable,
        points: np.ndarray,
        lags: Union[List, np.ndarray],
        custom_weights: np.ndarray = None
):
    """
    Calculates omnidirectional semivariance or covariance.

    Parameters
    ----------
    fn : function
        Covariance or semivariance func.

    points : numpy array
        ``[x, y, value]``

    lags : list or numpy array
        The list of lags.

    custom_weights : numpy array, optional

    Returns
    -------
    : numpy array
        Ordered semivariances or covariances
        ``[lag, value, number of point pairs]``
    """

    coordinates = points[:, :-1]

    distances = point_distance(
        points=coordinates,
        other=coordinates
    )

    omnidirectional_values = []

    def _get(lag_idx):
        omnidirectional_values.append(
            _calc_omnidirectional_values(
                fn=fn,
                points=points,
                distances=distances,
                lags=lags,
                lag_idx=lag_idx,
                custom_weights=custom_weights
            )
        )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for idx in range(len(lags)):
            futures.append(
                executor.submit(
                    _get, idx
                )
            )
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                raise e

    omnidirectional_values = np.array(omnidirectional_values)
    sorted_omnidirectional_values = omnidirectional_values[
        omnidirectional_values[:, 0].argsort()
    ]

    return sorted_omnidirectional_values


def _clean_cloud_data(ds: List):
    """
    Function cleans and sorts data with the semivariogram point clouds.

    Parameters
    ----------
    ds : List
        ``[lag, [point cloud]]``

    Returns
    -------
    : OrderedDict
        ``{lag: [point cloud]}``
    """
    # sort
    sort_ds = sorted(ds, key=itemgetter(0))
    ds = OrderedDict()
    for rec in sort_ds:
        ds[rec[0]] = rec[1]
    return ds


def _calc_omnidirectional_cloud(points: np.ndarray,
                                distances: np.ndarray,
                                lags: np.ndarray,
                                lag_idx: int,
                                raise_when_no_neighbors: bool = True):
    """
    Function calculates covariance for a given lag, returning lag,
    covariance and number of point pairs in range.

    Parameters
    ----------
    points : numpy array
        ``[x, y, value]``

    distances : numpy array
        Distances from each point to other points.

    lags : list or numpy array
        The list of lags.

    lag_idx : int
        Current lag index.

    raise_when_no_neighbors : bool, default = True
        Raise error when no neighbors are selected for a given lag.

    Returns
    -------
    : Tuple
        ``(lag, semivariances, number of point pairs within a lag)``

    """
    if lag_idx == 0:
        current_lag = lags[lag_idx]
        previous_lag = 0
    else:
        current_lag = lags[lag_idx]
        previous_lag = lags[lag_idx - 1]

    current_lag = float(current_lag)  # type checking
    distances_in_range = select_values_in_range(distances,
                                                current_lag,
                                                previous_lag)

    if len(distances_in_range[0]) == 0:
        if lag_idx == 0:
            return current_lag, [], 0
        else:
            if raise_when_no_neighbors:
                msg = f'There are no neighbors for a lag {current_lag},' \
                      f'the process has been stopped.'
                raise RuntimeError(msg)
            else:
                return current_lag, [], 0
    else:
        vals_0 = points[distances_in_range[0], 2]
        vals_h = points[distances_in_range[1], 2]

        value = np.square(vals_0 - vals_h)

        return current_lag, value


def _calc_omnidirectional_values(fn: Callable,
                                 points: np.ndarray,
                                 distances: np.ndarray,
                                 lags: np.ndarray,
                                 lag_idx: int,
                                 custom_weights=None):
    """
    Function calculates covariance for a given lag, returning lag,
    covariance and number of point pairs in range.

    Parameters
    ----------
    fn : function
        Covariance or semivariance func.

    points : numpy array
        ``[x, y, value]``

    distances : numpy array
        Distances from each point to other points.

    lags : list or numpy array
        The list of lags.

    lag_idx : int
        Current lag index.

    custom_weights : numpy array, optional


    Returns
    -------
    : Tuple
        ``(lag, co- or semi-variance, number of point pairs within a lag)``

    """
    if lag_idx == 0:
        current_lag = lags[lag_idx]
        previous_lag = 0
    else:
        current_lag = lags[lag_idx]
        previous_lag = lags[lag_idx - 1]

    current_lag = float(current_lag)  # type checking
    distances_in_range = select_values_in_range(distances,
                                                current_lag,
                                                previous_lag)

    if len(distances_in_range[0]) == 0:
        if lag_idx == 0:
            return current_lag, 0, 0
        else:
            return current_lag, np.nan, np.nan
    else:
        vals_0 = points[distances_in_range[0], 2]
        vals_h = points[distances_in_range[1], 2]

        if custom_weights is None:
            value = fn(vals_0, vals_h)
        else:
            # TODO: the same for covariance
            value = weight_experimental_semivariance(
                weights=custom_weights,
                distances_in_range=distances_in_range,
                vals_0=vals_0,
                vals_h=vals_h
            )

        return current_lag, value, len(vals_0)
