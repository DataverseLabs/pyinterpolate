"""
Distance calculation functions.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Dict, Union, Hashable

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm


def _calc_b2b_dist_from_dataframe(
        ps_blocks: Union[pd.DataFrame, gpd.GeoDataFrame],
        lon_col_name: Union[str, Hashable],
        lat_col_name: Union[str, Hashable],
        val_col_name: Union[str, Hashable],
        block_id_col_name: Union[str, Hashable],
        verbose=False
) -> pd.DataFrame:
    r"""
    Function calculates distances between the blocks' point supports.

    Parameters
    ----------
    ps_blocks : Union[pd.DataFrame, gpd.GeoDataFrame]
        DataFrame with point supports and block indexes.

    lon_col_name : Union[str, Hashable]
        Longitude or x coordinate.

    lat_col_name : Union[str, Hashable]
        Latitude or y coordinate.

    val_col_name : Union[str, Hashable]
        The point support values column.

    block_id_col_name : Union[str, Hashable]
        Column with block names / indexes.

    verbose : bool, default = False
        Show progress bar.

    Returns
    -------
    block_distances : DataFrame
        Indexes and columns are block indexes, cells are distances.

    """
    calculated_pairs = set()
    unique_blocks = list(ps_blocks[block_id_col_name].unique())

    col_set = [lon_col_name, lat_col_name, val_col_name]

    results = []

    for block_i in tqdm(unique_blocks, disable=not verbose):
        for block_j in unique_blocks:
            # Check if it was estimated
            if not (block_i, block_j) in calculated_pairs:
                if block_i == block_j:
                    results.append([block_i, block_j, 0])
                else:
                    i_value = ps_blocks[
                        ps_blocks[block_id_col_name] == block_i
                    ]
                    j_value = ps_blocks[
                        ps_blocks[block_id_col_name] == block_j
                    ]
                    value = _calculate_block_to_block_distance(
                        i_value[col_set].to_numpy(),
                        j_value[col_set].to_numpy()
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


# noinspection PyUnresolvedReferences
def _calc_b2b_dist_from_ps(ps_blocks: 'PointSupport', verbose=False) -> Dict:
    r"""
    Function calculates distances between the blocks' point supports.

    Parameters
    ----------
    ps_blocks : PointSupport
        PointSupport object with parsed blocks and their respective point
        supports.

    verbose : bool, default = False
        Show progress bar.

    Returns
    -------
    block_distances : DataFrame
        Indexes and columns are block indexes, cells are distances.
    """
    block_distances = _calc_b2b_dist_from_dataframe(
        ps_blocks=ps_blocks.point_support,
        lon_col_name=ps_blocks.lon_col_name,
        lat_col_name=ps_blocks.lat_col_name,
        val_col_name=ps_blocks.value_column_name,
        block_id_col_name=ps_blocks.point_support_blocks_index_name,
        verbose=verbose
    )

    return block_distances


# noinspection PyUnresolvedReferences
def calc_block_to_block_distance(
        ps_blocks: Union[pd.DataFrame, 'PointSupport'],
        lon_col_name=None,
        lat_col_name=None,
        val_col_name=None,
        block_id_col_name=None,
        verbose=False
) -> pd.DataFrame:
    r"""
    Function calculates distances between the blocks' point supports.

    Parameters
    ----------
    ps_blocks : Union[pd.DataFrame, PointSupport]
        The point support of polygons.
          * ``DataFrame``, then ``lon_col_name``, ``lat_col_name``,
            ``val_col_name``, ``block_id_col_name`` parameters must be
            provided,
          * ``PointSupport``.

    lon_col_name : optional
        Longitude or x coordinate.

    lat_col_name : optional
        Latitude or y coordinate.

    val_col_name : optional
        The point support values column.

    block_id_col_name : optional
        Column with block names / indexes.

    verbose : bool, default = False
        Show progress bar.

    Returns
    -------
    block_distances : DataFrame
        Indexes and columns representing unique blocks, values in cells
        are weighted distances between blocks.

    Raises
    ------
    AttributeError
        Blocks are provided as DataFrame but column names were not given.
    """

    if isinstance(ps_blocks, pd.DataFrame):
        if (
                (lon_col_name is None) or
                (lat_col_name is None) or
                (val_col_name is None) or
                (block_id_col_name is None)
        ):
            raise AttributeError(
                'Please provide the required column names '
                'if you pass DataFrame into the function'
            )
        block_distances = _calc_b2b_dist_from_dataframe(
            ps_blocks,
            lon_col_name=lon_col_name,
            lat_col_name=lat_col_name,
            val_col_name=val_col_name,
            block_id_col_name=block_id_col_name,
            verbose=verbose
        )
    else:
        block_distances = _calc_b2b_dist_from_ps(ps_blocks, verbose=verbose)

    return block_distances


def _calculate_block_to_block_distance(ps_block_1: np.ndarray,
                                       ps_block_2: np.ndarray) -> float:
    r"""
    Function calculates distance between two blocks' point supports.

    Parameters
    ----------
    ps_block_1 : numpy array
        Point support of the first block.

    ps_block_2 : numpy array
        Point support of the second block.

    Returns
    -------
    weighted_distances : float
        Weighted point-support distance between blocks.

    Notes
    -----
    The weighted distance between blocks is derived from the equation given
    in publication [1] from References. This distance is weighted by

    References
    ----------
    .. [1] Goovaerts, P. Kriging and Semivariogram Deconvolution in the
           Presence of Irregular Geographical Units.
           Math Geosci 40, 101–128 (2008).
           https://doi.org/10.1007/s11004-007-9129-1

    TODO
    ----
    * Add references equation to special part of the documentation.
    """

    a_shape = ps_block_1.shape[0]
    b_shape = ps_block_2.shape[0]
    ax = ps_block_1[:, 0].reshape(1, a_shape)
    bx = ps_block_2[:, 0].reshape(b_shape, 1)
    dx = ax - bx
    ay = ps_block_1[:, 1].reshape(1, a_shape)
    by = ps_block_2[:, 1].reshape(b_shape, 1)
    dy = ay - by
    aval = ps_block_1[:, -1].reshape(1, a_shape)
    bval = ps_block_2[:, -1].reshape(b_shape, 1)
    w = aval * bval

    dist = np.sqrt(dx ** 2 + dy ** 2, dtype=float, casting='unsafe')

    wdist = dist * w
    distances_sum = np.sum(wdist) / np.sum(w)
    return distances_sum


def select_neighbors_in_range(data: pd.DataFrame,
                              current_lag: float,
                              previous_lag: float):
    """
    Function selects the neighbors of each block within a range given by
    previous and current lags.

    Parameters
    ----------
    data : DataFrame
        Distances between blocks, columns and rows are block indexes.

    current_lag : float
        Actual maximum distance.

    previous_lag : float
        Previous maximum distance.

    Returns
    -------
    neighbors : Dict
        block id: [list of neighbors within a range
        given by previous and current lags]
    """

    cols = list(data.columns)
    neighbors = data.reset_index(names='block_i')
    neighbors = neighbors.melt(id_vars='block_i', value_vars=cols)

    neighbors = neighbors[
        (neighbors > previous_lag) & (neighbors <= current_lag)
    ]
    dneighbors = neighbors.dropna()

    if len(dneighbors) == 0:
        return {}
    else:
        pneighbors = dneighbors.drop(columns='value')
        other_neighbors_col = pneighbors.columns[-1]
        pneighbors = pneighbors.groupby('block_i')[
            other_neighbors_col
        ].apply(list)

        output = pneighbors.to_dict()

        return output
