"""
Data transforming functions.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Dict, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from pyinterpolate.processing.preprocessing.blocks import PointSupport, Blocks


def block_arr_to_dict(arr: np.ndarray):
    """Function transforms block array into a dict.

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
                            idx_col='index',
                            x_col='x',
                            y_col='y',
                            value_col='ds') -> Dict:
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


def get_areal_values_from_agg(
        aggregated_data: Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray],
        indexes=None) -> np.ndarray:
    """

    Parameters
    ----------
    aggregated_data : Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray]
                      Blocks with aggregated data.
                      * Blocks: Blocks() class object.
                      * GeoDataFrame and DataFrame must have columns: centroid.x, centroid.y, ds, index.
                        Geometry column with polygons is not used and optional.
                      * numpy array: [[block index, centroid x, centroid y, value]].

    indexes : Iterable (optional)
              List of indexes that are included in the output.

    Returns
    -------
    : numpy array
        [values]
    """
    if isinstance(aggregated_data, Blocks):
        val = aggregated_data.value_column_name
        ds = aggregated_data.data
        if indexes is not None:
            idx_col = aggregated_data.index_column_name
            ds.set_index(idx_col, inplace=True)
            ds = ds.loc[indexes]
        ds = ds[val].values
    elif isinstance(aggregated_data, pd.DataFrame) or isinstance(aggregated_data, gpd.GeoDataFrame):

        expected_cols = ['ds']

        if indexes is not None:
            expected_cols.append('index')

        if not set(expected_cols).issubset(set(aggregated_data.columns)):
            raise KeyError(f'Given dataframe doesnt have all expected columns {expected_cols}. '
                           f'It has {aggregated_data.columns} instead.')

        ds = aggregated_data[expected_cols]

        if indexes is not None:
            ds.set_index('index', inplace=True)
            ds = ds.loc[indexes]

        ds = ds['ds'].values

    elif isinstance(aggregated_data, np.ndarray):
        if indexes is not None:
            ds = []
            for idx in indexes:
                ags = aggregated_data[aggregated_data[:, 0] == idx]
                ds.append(ags[0][3])
            ds = np.array(ds)
        else:
            ds = aggregated_data[:, -1]
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

    cls = [point_support.x_col, point_support.y_col, point_support.value_column]
    d = {}
    for _id in block_keys:
        d[_id] = point_support.point_support[
                     point_support.point_support[point_support.block_index_column] == _id
                     ][cls].values
    return d


def sem_to_cov(semivariances, sill) -> np.ndarray:
    """
    Function transforms semivariances into a covariances.

    Parameters
    ----------
    semivariances : Iterable

    sill : float

    Returns
    -------
    covariances : numpy array
    """

    if isinstance(semivariances, np.ndarray):
        return sill - semivariances

    return sill - np.asarray(semivariances)


def transform_ps_to_dict(ps: Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport],
                         df_x_col_name='x',
                         df_y_col_name='y',
                         df_value_col_name='ds',
                         df_index_col_name='index') -> Dict:
    """

    Parameters
    ----------
    ps : Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport]
         * Dict: {block id: [[point x, point y, value]]}
         * numpy array: [[block id, x, y, value]]
         * DataFrame and GeoDataFrame: columns={x, y, ds, index}
         * PointSupport

    df_x_col_name : str, default='x'
        If DataFrame or GeoDataFrame is passed then this parameter represents the name of a column x (longitude).

    df_y_col_name : str, default='y'
        If DataFrame or GeoDataFrame is passed then this parameter represents the name of a column y (latitude).

    df_value_col_name : str, default='ds'
        If DataFrame or GeoDataFrame is passed then this parameter represents the name of a column with point support
        values.

    df_index_col_name : str, default='index'
        If DataFrame or GeoDataFrame is passed then this parameter represents the name of a column that can represent
        the point support index.

    Returns
    -------
    : Dict
        Point Support as a Dict: {block id: [[point x, point y, value]]}
    """
    if isinstance(ps, PointSupport):
        return point_support_to_dict(ps)
    elif isinstance(ps, pd.DataFrame) or isinstance(ps, gpd.GeoDataFrame):
        expected_cols = {df_x_col_name, df_y_col_name, df_value_col_name, df_index_col_name}

        if not expected_cols.issubset(set(ps.columns)):
            raise KeyError(f'Given dataframe doesnt have all expected columns {expected_cols}. '
                           f'It has {ps.columns} instead. Set those columns as function parameters.')
        return block_dataframe_to_dict(ps)
    elif isinstance(ps, np.ndarray):
        return block_arr_to_dict(ps)
    elif isinstance(ps, Dict):
        return ps
    else:
        raise TypeError(f'Blocks data type {type(ps)} not recognized. You may use PointSupport,'
                        f' Geopandas GeoDataFrame, Pandas DataFrame or numpy array. See docs.')


def transform_blocks_to_numpy(blocks: Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Function transforms blocks data into numpy array.

    Parameters
    ----------
    blocks : Union[Blocks, gpd.GeoDataFrame, pd.DataFrame, np.ndarray]
             Blocks with aggregated data.
             * Blocks: Blocks() class object.
             * GeoDataFrame and DataFrame must have columns: centroid.x, centroid.y, ds, index.
               Geometry column with polygons is not used and optional.
             * numpy array: [[block index, centroid x, centroid y, value]].

    Returns
    -------
    bvalues : numpy array
              Blocks transformed to numpy array [[block index, centroid x, centroid y, value]].
    """
    if isinstance(blocks, Blocks):
        bvalues = blocks.data[[blocks.index_column_name, blocks.cx, blocks.cy, blocks.value_column_name]].values
        return bvalues
    elif isinstance(blocks, pd.DataFrame) or isinstance(blocks, gpd.GeoDataFrame):
        expected_cols = {'centroid.x', 'centroid.y', 'ds', 'index'}

        if not expected_cols.issubset(set(blocks.columns)):
            raise KeyError(f'Given dataframe doesnt have all expected columns {expected_cols}. '
                           f'It has {blocks.columns} instead.')

        bvalues = blocks.data[['index', 'centroid.x', 'centroid.y', 'ds']].values
        return bvalues
    else:
        raise TypeError(f'Blocks data type {type(blocks)} not recognized. You may use Blocks,'
                        f' Geopandas GeoDataFrame, Pandas DataFrame or numpy array. See docs.')
