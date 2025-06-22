"""
Functions for calculating the semivariances between blocks' point supports.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Dict, Collection, Union, Hashable
import numpy as np
import pandas as pd

from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.distance.point import point_distance, select_values_in_range
from pyinterpolate.semivariogram.lags.lags import get_current_and_previous_lag
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram


def average_block_to_block_semivariances(semivariances_array: np.ndarray,
                                         lags: np.ndarray) -> np.ndarray:
    r"""
    Function averages block to block semivariances over specified lags.

    Parameters
    ----------
    semivariances_array : numpy array
        ``[lag, semivariance, number of point pairs between blocks]``

    lags : numpy array
        Ordered lags.

    Returns
    -------
    averaged : numpy array
        ``[lag, mean semivariance, number of point pairs in range]``
    """

    averaged = []
    distances = semivariances_array[:, 0]
    for idx, lag in enumerate(lags):
        current_lag, previous_lag = get_current_and_previous_lag(
            lag_idx=idx,
            lags=lags
        )
        distances_in_range = select_values_in_range(distances,
                                                    current_lag,
                                                    previous_lag)
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


def block_pair_semivariance(block_a: np.ndarray,
                            block_b: np.ndarray,
                            semivariogram_model: TheoreticalVariogram):
    r"""
    Function calculates average semivariance between the blocks'
    points supports based on predicted semivariance (using given model).

    Parameters
    ----------
    block_a : Collection
        Block A coordinates of point support points ``[x, y]``.

    block_b : Collection
        Block B coordinates of point support points ``[x, y]``.

    semivariogram_model : TheoreticalVariogram
        Fitted theoretical variogram model.

    Returns
    -------
    semivariance_between_blocks : float
        The average semivariance between blocks' point supports.
    """
    distances_between_points = point_distance(block_a, block_b)

    if not isinstance(distances_between_points, np.ndarray):
        distances_between_points = np.array(distances_between_points)

    predictions = semivariogram_model.predict(
        distances_between_points.flatten()
    )
    semivariance_between_blocks = np.mean(predictions)

    return semivariance_between_blocks


def calculate_block_to_block_semivariance(
        point_support: PointSupport,
        block_to_block_distances: pd.DataFrame,
        semivariogram_model: TheoreticalVariogram
) -> Dict:
    r"""
    Function calculates semivariance between blocks based on their
    point support and weighted distance between block coordinates.

    Parameters
    ----------
    point_support : PointSupport

    block_to_block_distances : Dict
        ``{block id : [distances to other blocks in order of keys]}``

    semivariogram_model : TheoreticalVariogram
        Fitted variogram model.

    Returns
    -------
    semivariances_b2b : Dict
        Keys are tuples - (block a index, block b index), and values are
        ``[distance, semivariance, number of point pairs between blocks]``
    """
    # Run
    blocks_ids = block_to_block_distances.index.tolist()
    lon_col = point_support.lon_col_name
    lat_col = point_support.lat_col_name
    block_idx = point_support.point_support_blocks_index_name

    semivariances_b2b = {}

    for first in blocks_ids:
        for second in blocks_ids:
            pair = (first, second)
            rev_pair = (second, first)

            if first == second:
                semivariances_b2b[pair] = [0, 0, 0]
                semivariances_b2b[rev_pair] = [0, 0, 0]
            else:
                if pair not in semivariances_b2b:
                    distance = block_to_block_distances.loc[first, second]

                    # Select set of points to calculate block-pair semivariance
                    ps = point_support.point_support
                    a_block_points = ps[
                        ps[block_idx] == first
                    ][[lon_col, lat_col]].to_numpy()
                    b_block_points = ps[
                        ps[block_idx] == second
                    ][[lon_col, lat_col]].to_numpy()
                    no_of_p_pairs = len(a_block_points) * len(b_block_points)

                    # Calculate semivariance between blocks
                    semivariance = block_pair_semivariance(a_block_points,
                                                           b_block_points,
                                                           semivariogram_model)

                    semivariances_b2b[pair] = [
                        distance,
                        semivariance,
                        no_of_p_pairs
                    ]
                    semivariances_b2b[rev_pair] = [
                        distance,
                        semivariance,
                        no_of_p_pairs
                    ]

    return semivariances_b2b


def weighted_avg_point_support_semivariances(
        semivariogram_model: TheoreticalVariogram,
        distances_between_neighboring_point_supports: pd.DataFrame,
        index_col: Union[Hashable, str],
        val1_col: Union[Hashable, str],
        val2_col: Union[Hashable, str],
        dist_col: Union[Hashable, str]) -> pd.Series:
    r"""
    Function calculates semivariances between given point support distances.

    Parameters
    ----------
    semivariogram_model : TheoreticalVariogram
        Fitted model.

    distances_between_neighboring_point_supports : DataFrame
        Block pairs and their respective point support points and
        distances between those points.

    index_col : Union[Hashable, str]
        The name of the column with block indexes (tuple with two blocks).

    val1_col : Union[Hashable, str]
        The name of the column with the first point value.

    val2_col : Union[Hashable, str]
        The name of the column with the second point value.

    dist_col : Union[Hashable, str]
        The name of the column with distances between points in point
        supports.

    Returns
    -------
    : pandas Series
        Weighted semivariance between point supports
        ``{index_col: semivariance}``

    Notes
    -----

    Weighted semivariance is calculated as:

    (1)

    $$\gamma_{v_{i}, v_{j}}
        =
        \frac{1}{\sum_{s}^{P_{i}} \sum_{s'}^{P_{j}} w_{ss'}} *
            \sum_{s}^{P_{i}} \sum_{s'}^{P_{j}} w_{ss'} *
            \gamma(u_{s}, u_{s'})$$

    where:
    - $w_{ss'}$ - product of point-support weights from block a and block b.
    - $\gamma(u_{s}, u_{s'})$ - semivariance between point-supports of
      block a and block b.
    """

    ds = distances_between_neighboring_point_supports.copy()

    ds['gamma'] = semivariogram_model.predict(ds[dist_col].to_numpy())

    # custom_weights
    ds['custom_weights'] = ds[val1_col] * ds[val2_col]
    ds['gamma_weights'] = ds['gamma'] * ds['custom_weights']

    # groupby
    gs = ds.groupby(index_col)[['custom_weights', 'gamma_weights']].sum()

    # Calculate weighted semivariance
    weighted_semivariance = gs['gamma_weights'] / gs['custom_weights']

    return weighted_semivariance
