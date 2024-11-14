"""
Distance calculation functions.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Dict, Union

import geopandas as gpd
import numpy as np
import pandas as pd


def _calc_b2b_dist_from_dataframe(blocks: Union[pd.DataFrame, gpd.GeoDataFrame], lon, lat, val, bidx) -> pd.DataFrame:
    """Function calculates distances between blocks.

    Parameters
    ----------
    blocks : Union[pd.DataFrame, gpd.GeoDataFrame]

    lon
        Longitude or x coordinate.

    lat
        Latitude or y coordinate.

    val
        The point support values column.

    bidx
        Column with block names / indexes.

    Returns
    -------
    block_distances : DataFrame
        Indexes and columns are block indexes, cells are distances.

    """
    calculated_pairs = set()
    unique_blocks = list(blocks[bidx].unique())

    results = []

    for block_i in unique_blocks:
        for block_j in unique_blocks:
            # Check if it was estimated
            if not (block_i, block_j) in calculated_pairs:
                if block_i == block_j:
                    results.append([block_i, block_j, 0])
                else:
                    i_value = blocks[blocks[bidx] == block_i]
                    j_value = blocks[blocks[bidx] == block_j]
                    value = _calculate_block_to_block_distance(
                        i_value[[lon, lat, val]].to_numpy(),
                        j_value[[lon, lat, val]].to_numpy()
                    )
                    results.append([block_i, block_j, value])
                    results.append([block_j, block_i, value])
                    calculated_pairs.add((block_i, block_j))
                    calculated_pairs.add((block_j, block_i))

    # Create output dataframe
    df = pd.DataFrame(data=results, columns=['block_i', 'block_j', 'z'])
    df = df.pivot_table(
        values='z',
        index='block_i',
        columns='block_j'
    )

    # sort
    df = df.reindex(columns=unique_blocks)
    df = df.reindex(index=unique_blocks)

    return df


def _calc_b2b_dist_from_ps(blocks: 'PointSupport') -> Dict:
    """Function calculates distances between blocks.

    Parameters
    ----------
    blocks : PointSupport

    Returns
    -------
    block_distances : DataFrame
        Indexes and columns are block indexes, cells are distances.
    """
    block_distances = _calc_b2b_dist_from_dataframe(
        blocks=blocks.point_support,
        lon=blocks.lon_col_name,
        lat=blocks.lat_col_name,
        val=blocks.value_column_name,
        bidx=blocks.point_support_blocks_index_name
    )

    return block_distances


def calc_block_to_block_distance(blocks: Union[pd.DataFrame, 'PointSupport'],
                                 lon_col_name=None,
                                 lat_col_name=None,
                                 val_col_name=None,
                                 block_index_col_name=None) -> pd.DataFrame:
    """Function calculates distances between blocks.

    Parameters
    ----------
    blocks : Union[pd.DataFrame, PointSupport]
        The point support of polygons.
          * ``DataFrame``, then ``x_col``, ``y_col``, ``val_col``, ``block_index_col`` parameters must be provided,
          * ``PointSupport``.

    lon_col_name : optional
        Longitude or x coordinate.

    lat_col_name : optional
        Latitude or y coordinate.

    val_col_name : optional
        The point support values column.

    block_index_col_name : optional
        Column with block names / indexes.

    Returns
    -------
    block_distances : DataFrame
        Indexes and columns representing blocks (blocks id), cells are distances.

    Raises
    ------
    AttributeError
        Blocks are provided as DataFrame but column names were not given.
    """

    if isinstance(blocks, pd.DataFrame):
        if (
                (lon_col_name is None) or
                (lat_col_name is None) or
                (val_col_name is None) or
                (block_index_col_name is None)
        ):
            raise AttributeError(
                'Please provide the required column names if you pass DataFrame into the function'
            )
        block_distances = _calc_b2b_dist_from_dataframe(blocks,
                                                        lon=lon_col_name,
                                                        lat=lat_col_name,
                                                        val=val_col_name,
                                                        bidx=block_index_col_name)
    else:
        block_distances = _calc_b2b_dist_from_ps(blocks)

    return block_distances


def _calculate_block_to_block_distance(block_1: np.ndarray, block_2: np.ndarray) -> float:
    """Function calculates distance between two blocks based on how they are divided (into the point support grid).

    Parameters
    ----------
    block_1 : numpy array

    block_2 : numpy array

    Returns
    -------
    weighted_distances : float
                         Weighted distance between blocks.

    Notes
    -----
    The weighted distance between blocks is derived from the equation:

    $$d(v_{a}, v_{b})=\frac{1}{\sum_{s=1}^{P_{a}} \sum_{s'=1}^{P_{b}} n(u_{s}) n(u_{s'})} *
        \sum_{s=1}^{P_{a}} \sum_{s'=1}^{P_{b}} n(u_{s})n(u_{s'})||u_{s}-u_{s'}||$$

    where:
    $P_{a}$ and $P_{b}$: number of points $u_{s}$ and $u_{s'}$ used to discretize the two units $v_{a}$ and $v_{b}$,
    $n(u_{s})$ and $n(u_{s'})$ - population size in the cells $u_{s}$ and $u_{s'}$.

    References
    ----------
    .. [1] Goovaerts, P. Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical Units.
           Math Geosci 40, 101–128 (2008). https://doi.org/10.1007/s11004-007-9129-1
    """

    a_shape = block_1.shape[0]
    b_shape = block_2.shape[0]
    ax = block_1[:, 0].reshape(1, a_shape)
    bx = block_2[:, 0].reshape(b_shape, 1)
    dx = ax - bx
    ay = block_1[:, 1].reshape(1, a_shape)
    by = block_2[:, 1].reshape(b_shape, 1)
    dy = ay - by
    aval = block_1[:, -1].reshape(1, a_shape)
    bval = block_2[:, -1].reshape(b_shape, 1)
    w = aval * bval

    dist = np.sqrt(dx ** 2 + dy ** 2, dtype=float, casting='unsafe')

    wdist = dist * w
    distances_sum = np.sum(wdist) / np.sum(w)
    return distances_sum
