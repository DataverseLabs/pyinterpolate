from typing import Dict, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from pyinterpolate.processing.preprocessing.blocks import PointSupport, Blocks


def block_arr_to_dict(arr: np.ndarray):
    """
    Function transforms block array into a dict.

    Parameters
    ----------
    arr : numpy array
        [block idx, x, y, value]

    Returns
    -------
    : Dict
        {block id: [[numpy array with x, y, value]]}
    """
    d = {}

    for unique_k in np.unique(arr[:, 0]):
        d[unique_k] = arr[arr[:, 0] == unique_k][:, 1:]

    return d


def block_dataframe_to_dict(block_df: Union[pd.DataFrame, gpd.GeoDataFrame],
                            idx_col='index', x_col='x', y_col='y', value_col='ds') -> Dict:
    """

    Parameters
    ----------
    block_df : Union[DataFrame, GeoDataFrame]

    idx_col : any, default='index'
        Index column name.

    x_col : any, default='x'
        X coordinates.

    y_col : any, default='y'
        Y coordinates.

    value_col : any, default='ds'
        Values.

    Returns
    -------
    : Dict
        {block id: [[numpy array with x, y, value]]}
    """
    d = {}
    for _id in block_df[idx_col].unique():
        d[_id] = block_df[block_df[idx_col] == _id][[x_col, y_col, value_col]].values

    return d


def get_areal_centroids_from_agg(
        aggregated_data: Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray]) -> np.ndarray:
    """

    Parameters
    ----------
    aggregated_data : Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray]
                      Blocks with aggregated data.
                      * Blocks: Blocks() class object.
                      * GeoDataFrame and DataFrame must have columns: centroid.x, centroid.y, ds, index.
                        Geometry column with polygons is not used and optional.
                      * numpy array: [[block index, centroid x, centroid y, value]].

    Returns
    -------
    : numpy array
        [[cx, cy, val]]
    """
    if isinstance(aggregated_data, Blocks):
        cx = aggregated_data.cx
        cy = aggregated_data.cy
        val = aggregated_data.value_column_name
        ds = aggregated_data.data
        ds = ds[[cx, cy, val]].values
    elif isinstance(aggregated_data, pd.DataFrame) or isinstance(aggregated_data, gpd.GeoDataFrame):

        expected_cols = ['centroid.x', 'centroid.y', 'ds']
        if not set(expected_cols).issubset(set(aggregated_data.columns)):
            raise KeyError(f'Given dataframe doesnt have all expected columns {expected_cols}. '
                           f'It has {aggregated_data.columns} instead.')

        ds = aggregated_data[expected_cols].values

    elif isinstance(aggregated_data, np.ndarray):
        ds = aggregated_data[:, 1:]
    else:
        raise TypeError(f'Blocks data type {type(aggregated_data)} not recognized. You may use Blocks,'
                        f' Geopandas GeoDataFrame, Pandas DataFrame or numpy array. See docs.')

    return ds


def point_support_to_dict(point_support: PointSupport) -> Dict:
    """
    Function transforms PointSupport into Dict.

    Parameters
    ----------
    point_support : PointSupport

    Returns
    -------
    : Dict
        {block id: [[numpy array with x, y, value]]}
    """
    block_keys = point_support.point_support[point_support.block_index_column].unique()

    cls = [point_support.block_index_column, point_support.x_col, point_support.y_col, point_support.value_column]
    d = {}
    for _id in block_keys:
        d[_id] = point_support.point_support[
                     point_support.point_support[point_support.block_index_column] == _id
                     ][cls].values[:, 1:]
    return d