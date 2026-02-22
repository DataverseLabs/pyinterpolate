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

    Notes
    -----
    The weighted distance between blocks is derived from the equation 3 given
    in publication [1] from References.

    .. math:: d(v_{a}, v_{b})=\frac{1}{\sum_{s=1}^{P_{a}} \sum_{s'=1}^{P_{b}} n(u_{s}) n(u_{s'})} * \sum_{s=1}^{P_{a}} \sum_{s'=1}^{P_{b}} n(u_{s})n(u_{s'})||u_{s}-u_{s'}||

    where:
      * :math:`P_{a}` and :math:`P_{b}`: number of points :math:`u_{s}`
        and :math:`u_{s'}` used to discretize the two units :math:`v_{a}`
        and :math:`v_{b}`
      * :math:`n(u_{s})` and :math:`n(u_{s'})` - population size in
        the cells :math:`u_{s}` and :math:`u_{s'}`

    References
    ----------
    .. [1] Goovaerts, P. Kriging and Semivariogram Deconvolution in the
           Presence of Irregular Geographical Units.
           Math Geosci 40, 101–128 (2008).
           https://doi.org/10.1007/s11004-007-9129-1

    Examples
    --------
    >>> import os
    >>> import geopandas as gpd
    >>> from pyinterpolate import (
    ...     calc_block_to_block_distance,
    ...     Blocks,
    ...     ExperimentalVariogram,
    ...     PointSupport,
    ...     TheoreticalVariogram
    ... )
    >>>
    >>>
    >>> FILENAME = 'cancer_data.gpkg'
    >>> LAYER_NAME = 'areas'
    >>> DS = gpd.read_file(FILENAME, layer=LAYER_NAME)
    >>> AREA_VALUES = 'rate'
    >>> AREA_INDEX = 'FIPS'
    >>> AREA_GEOMETRY = 'geometry'
    >>> PS_LAYER_NAME = 'points'
    >>> PS_VALUES = 'POP10'
    >>> PS_GEOMETRY = 'geometry'
    >>> PS = gpd.read_file(FILENAME, layer=PS_LAYER_NAME)
    >>>
    >>> CANCER_DATA = {
    ...    'ds': DS,
    ...    'index_column_name': AREA_INDEX,
    ...    'value_column_name': AREA_VALUES,
    ...    'geometry_column_name': AREA_GEOMETRY
    ... }
    >>> POINT_SUPPORT_DATA = {
    ...     'ps': PS,
    ...     'value_column_name': PS_VALUES,
    ...     'geometry_column_name': PS_GEOMETRY
    ... }
    >>> BLOCKS = Blocks(**CANCER_DATA)
    >>> indexes = BLOCKS.block_indexes
    >>>
    >>> PS = PointSupport(
    ...     points=POINT_SUPPORT_DATA['ps'],
    ...     ps_blocks=BLOCKS,
    ...     points_value_column=POINT_SUPPORT_DATA['value_column_name'],
    ...     points_geometry_column=POINT_SUPPORT_DATA['geometry_column_name']
    ... )
    >>> distances = calc_block_to_block_distance(PS)
    >>> print(distances.head())
                   42049.0        42039.0  ...       23009.0       23029.0
    42049.0       0.000000   48885.422652  ...  9.780781e+05  1.061496e+06
    42039.0   48885.422652       0.000000  ...  9.952927e+05  1.079528e+06
    42085.0   91350.566468   52240.566481  ...  1.030860e+06  1.115642e+06
    42073.0  120259.614440   78853.981609  ...  1.042438e+06  1.127643e+06
    42007.0  152110.979345  109955.408914  ...  1.055898e+06  1.141472e+06
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
