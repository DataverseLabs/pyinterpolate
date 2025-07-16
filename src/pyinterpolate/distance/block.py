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
from scipy.spatial.distance import cdist


def _calc_b2b_dist_from_dataframe(
        ps_blocks: Union[pd.DataFrame, gpd.GeoDataFrame],
        lon_col_name: Union[str, Hashable],
        lat_col_name: Union[str, Hashable],
        val_col_name: Union[str, Hashable],
        block_id_col_name: Union[str, Hashable]
) -> pd.DataFrame:
    """Fully vectorized version - fastest approach."""

    unique_blocks = ps_blocks[block_id_col_name].unique()
    n_blocks = len(unique_blocks)

    # Pre-compute block data once
    block_data = {}
    for block_id in unique_blocks:
        mask = ps_blocks[block_id_col_name] == block_id
        coords = ps_blocks.loc[mask, [lon_col_name, lat_col_name]].values
        weights = ps_blocks.loc[mask, val_col_name].values
        block_data[block_id] = (coords, weights)

    # Initialize distance matrix
    distance_matrix = np.zeros((n_blocks, n_blocks))

    # Compute distances for upper triangle only
    for i in range(n_blocks):
        for j in range(i, n_blocks):
            if i == j:
                distance_matrix[i, j] = 0.0
            else:
                coords_i, weights_i = block_data[unique_blocks[i]]
                coords_j, weights_j = block_data[unique_blocks[j]]

                # Use scipy's cdist for pairwise distances
                dist_matrix = cdist(coords_i, coords_j)
                weight_matrix = np.outer(weights_i, weights_j)

                distance = np.sum(dist_matrix * weight_matrix) / np.sum(
                    weight_matrix)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # Symmetric

    # Create DataFrame
    return pd.DataFrame(
        distance_matrix,
        index=unique_blocks,
        columns=unique_blocks
    )


# def _calc_b2b_dist_from_dataframe(
#         ps_blocks: Union[pd.DataFrame, gpd.GeoDataFrame],
#         lon_col_name: Union[str, Hashable],
#         lat_col_name: Union[str, Hashable],
#         val_col_name: Union[str, Hashable],
#         block_id_col_name: Union[str, Hashable],
#         verbose=False
# ) -> pd.DataFrame:
#     r"""
#     Function calculates distances between the blocks' point supports.
#
#     Parameters
#     ----------
#     ps_blocks : Union[pd.DataFrame, gpd.GeoDataFrame]
#         DataFrame with point supports and block indexes.
#
#     lon_col_name : Union[str, Hashable]
#         Longitude or x coordinate.
#
#     lat_col_name : Union[str, Hashable]
#         Latitude or y coordinate.
#
#     val_col_name : Union[str, Hashable]
#         The point support values column.
#
#     block_id_col_name : Union[str, Hashable]
#         Column with block names / indexes.
#
#     verbose : bool, default = False
#         Show progress bar.
#
#     Returns
#     -------
#     block_distances : DataFrame
#         Indexes and columns are block indexes, cells are distances.
#
#     """
#     calculated_pairs = set()
#     unique_blocks = list(ps_blocks[block_id_col_name].unique())
#
#     col_set = [lon_col_name, lat_col_name, val_col_name]
#
#     results = []
#
#     for block_i in tqdm(unique_blocks, disable=not verbose):
#         for block_j in unique_blocks:
#             # Check if it was estimated
#             if not (block_i, block_j) in calculated_pairs:
#                 if block_i == block_j:
#                     results.append([block_i, block_j, 0])
#                 else:
#                     i_value = ps_blocks[
#                         ps_blocks[block_id_col_name] == block_i
#                     ]
#                     j_value = ps_blocks[
#                         ps_blocks[block_id_col_name] == block_j
#                     ]
#                     value = _calculate_block_to_block_distance(
#                         i_value[col_set].to_numpy(),
#                         j_value[col_set].to_numpy()
#                     )
#                     results.append([block_i, block_j, value])
#                     results.append([block_j, block_i, value])
#                     calculated_pairs.add((block_i, block_j))
#                     calculated_pairs.add((block_j, block_i))
#
#     # Create output dataframe
#     df = pd.DataFrame(data=results, columns=['block_i', 'block_j', 'z'])
#     df = df.pivot_table(
#         values='z',
#         index='block_i',
#         columns='block_j'
#     )
#
#     # sort
#     df = df.reindex(columns=unique_blocks)
#     df = df.reindex(index=unique_blocks)
#
#     return df


# noinspection PyUnresolvedReferences
def _calc_b2b_dist_from_ps(ps_blocks: 'PointSupport') -> Dict:
    r"""
    Function calculates distances between the blocks' point supports.

    Parameters
    ----------
    ps_blocks : PointSupport
        PointSupport object with parsed blocks and their respective point
        supports.

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
        block_id_col_name=ps_blocks.point_support_blocks_index_name
    )

    return block_distances


# noinspection PyUnresolvedReferences
def calc_block_to_block_distance(
        ps_blocks: Union[pd.DataFrame, 'PointSupport'],
        lon_col_name=None,
        lat_col_name=None,
        val_col_name=None,
        block_id_col_name=None
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
            block_id_col_name=block_id_col_name
        )
    else:
        block_distances = _calc_b2b_dist_from_ps(ps_blocks)

    return block_distances


# def _calculate_block_to_block_distance(ps_block_1: np.ndarray,
#                                        ps_block_2: np.ndarray) -> float:
#     r"""
#     Function calculates distance between two blocks' point supports.
#
#     Parameters
#     ----------
#     ps_block_1 : numpy array
#         Point support of the first block.
#
#     ps_block_2 : numpy array
#         Point support of the second block.
#
#     Returns
#     -------
#     weighted_distances : float
#         Weighted point-support distance between blocks.
#
#     Notes
#     -----
#     The weighted distance between blocks is derived from the equation given
#     in publication [1] from References. This distance is weighted by
#
#     References
#     ----------
#     .. [1] Goovaerts, P. Kriging and Semivariogram Deconvolution in the
#            Presence of Irregular Geographical Units.
#            Math Geosci 40, 101–128 (2008).
#            https://doi.org/10.1007/s11004-007-9129-1
#
#     TODO
#     ----
#     * Add reference equation to the special part of the documentation.
#     """
#
#     a_shape = ps_block_1.shape[0]
#     b_shape = ps_block_2.shape[0]
#     ax = ps_block_1[:, 0].reshape(1, a_shape)
#     bx = ps_block_2[:, 0].reshape(b_shape, 1)
#     dx = ax - bx
#     ay = ps_block_1[:, 1].reshape(1, a_shape)
#     by = ps_block_2[:, 1].reshape(b_shape, 1)
#     dy = ay - by
#     aval = ps_block_1[:, -1].reshape(1, a_shape)
#     bval = ps_block_2[:, -1].reshape(b_shape, 1)
#     w = aval * bval
#
#     dist = np.sqrt(dx ** 2 + dy ** 2, dtype=float, casting='unsafe')
#
#     wdist = dist * w
#     distances_sum = np.sum(wdist) / np.sum(w)
#     return distances_sum


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
        block index: [list of neighbor indexes within a range
        given by previous and current lags]
    """

    neighbors = data.reset_index()
    neighbors = neighbors.melt(id_vars=neighbors.columns[0], value_vars=neighbors.columns[1:])

    neighbors = neighbors[
        (neighbors['value'] > previous_lag) & (neighbors['value'] <= current_lag)
    ]
    dneighbors = neighbors.dropna()

    if len(dneighbors) == 0:
        return {}
    else:
        pneighbors = dneighbors.drop(columns='value')
        other_neighbors_col = pneighbors.columns[-1]
        pneighbors = pneighbors.groupby(pneighbors.columns[0])[
            other_neighbors_col
        ].apply(list)

        output = pneighbors.to_dict()

        return output
