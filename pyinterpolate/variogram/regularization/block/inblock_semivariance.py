"""
Functions for calculating the inblock point-support semivariance.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Dict, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from pyinterpolate.distance.distance import calc_point_to_point_distance
from pyinterpolate.processing.preprocessing.blocks import PointSupport
from pyinterpolate.variogram.theoretical.semivariogram import TheoreticalVariogram


def inblock_semivariance(points_of_block: np.ndarray, variogram_model: TheoreticalVariogram) -> float:
    """
    Function calculates inblock semivariance.

    Parameters
    ----------
    points_of_block : numpy array

    variogram_model : TheoreticalVariogram

    Returns
    -------
    average_block_semivariance : float

    """
    number_of_points_within_block = len(points_of_block)  # P
    p = number_of_points_within_block * number_of_points_within_block  # P^2

    distances_between_points = calc_point_to_point_distance(points_of_block[:, :-1])  # Matrix of size PxP
    flattened_distances = distances_between_points.flatten()
    semivariances = variogram_model.predict(flattened_distances)

    average_block_semivariance = np.sum(semivariances) / p
    return average_block_semivariance


def calculate_inblock_semivariance(point_support: Union[Dict, PointSupport, gpd.GeoDataFrame, pd.DataFrame, np.ndarray],
                                   variogram_model: TheoreticalVariogram) -> Dict:
    """
    Method calculates inblock semivariance of a given areas.


    Parameters
    ----------
    point_support : geopandas GeoDataFrame | Point Support | numpy array
                    Point support data. It can be provided:
                        - directly as PointSupport object,
                        - GeoDataFrame | DataFrame (then DataFrame must have columns: 'ds' - values, 'x_col' - point
                          geometry x, 'y_col' - point geometry y, 'index' - block indexes,
                        - numpy array [block index, coordinate x, coordinate y, value],
                        - Dict: {block id: [[point x, point y, value]]}.

    variogram_model : TheoreticalVariogram
                      Modeled variogram fitted to the areal data.

    Returns
    -------
    inblock_semivariances : Dict
                            {area id: the average inblock semivariance}

    Notes
    -----
    $$\gamma(v, v) = \frac{1}{P^{2}} * \sum_{s}^{P} \sum_{s'}^{P} \gamma(u_{s}, u_{s}')$$

        where:
        - $\gamma(v, v)$ is the average semivariance within a block,
        - $P$ is a number of points used to discretize the block $v$,
        - $u_{s}$ is a point u within the block $v$,
        - $\gamma(u_{s}, u_{s}')$ is a semivariance between point $u_{s}$ and $u_{s}'$ inside the block $v$.
    """

    # TODO: It seems that multiprocessing gives the best results for point support matrices between
    #       10^2x10^2:10^4x10^4. It must be investigated further in the future!

    if isinstance(point_support, PointSupport):
        inblock_semivariances = _calculate_inblock_semivariance_from_point_support_class(point_support,
                                                                                         variogram_model)
    elif isinstance(point_support, gpd.GeoDataFrame) or isinstance(point_support, pd.DataFrame):
        inblock_semivariances = _calculate_inblock_semivariance_from_dataframe(point_support,
                                                                               variogram_model)
    elif isinstance(point_support, np.ndarray):
        inblock_semivariances = _calculate_inblock_semivariance_from_numpy_array(point_support,
                                                                                 variogram_model)
    elif isinstance(point_support, Dict):
        inblock_semivariances = _calculate_inblock_semivariance_from_dict(point_support, variogram_model)
    else:
        raise TypeError(f'Point support type {type(point_support)} not recognized. You may use PointSupport,'
                        f' Geopandas GeoDataFrame, Pandas DataFrame or numpy array. See docs.')

    return inblock_semivariances.copy()


def _calculate_inblock_semivariance_from_dataframe(point_support: Union[gpd.GeoDataFrame, pd.DataFrame],
                                                   variogram_model: TheoreticalVariogram):
    """
    Method calculates inblock semivariance of a given areas from the GeoDataFrame object. GeoDataFrame must have
    columns: 'ds' - values, 'x_col' - point geometry x, 'y_col' - point geometry y, 'index' - block indexes.


    Parameters
    ----------
    point_support : Union[gpd.GeoDataFrame, pd.DataFrame]
                    Columns: x, y, ds, index

    variogram_model : TheoreticalVariogram
                      Modeled variogram fitted to the areal data.

    Returns
    -------
    inblock_semivariances : Dict
                            {area id: the average inblock semivariance}
    """

    expected_cols = {'x_col', 'y_col', 'ds', 'index'}
    if not expected_cols.issubset(set(point_support.columns)):
        raise KeyError(f'Given dataframe doesnt have all expected columns {expected_cols}. '
                       f'It has {point_support.columns} instead.')

    inblock_semivariances = {}
    unique_areas = point_support['index'].unique()

    for unique_area in unique_areas:
        data_points = point_support[point_support['index'] == unique_area]

        data_points = data_points[
            ['x_col', 'y_col', 'ds']
        ].values

        inblock = inblock_semivariance(data_points, variogram_model)
        inblock_semivariances[unique_area] = inblock

    return inblock_semivariances


def _calculate_inblock_semivariance_from_dict(point_support: Dict, variogram_model: TheoreticalVariogram):
    """
    Method calculates inblock semivariance of a given areas from the Dict.


    Parameters
    ----------
    point_support : Dict
                    {area id: [[x, y, value]]}

    variogram_model : TheoreticalVariogram
                      Modeled variogram fitted to the areal data.

    Returns
    -------
    inblock_semivariances : Dict
                            {area id: the average inblock semivariance}
    """

    inblock_semivariances = {}
    unique_areas = list(point_support.keys())

    for unique_area in unique_areas:
        data_points = point_support[unique_area]

        inblock = inblock_semivariance(data_points, variogram_model)
        inblock_semivariances[unique_area] = inblock

    return inblock_semivariances


def _calculate_inblock_semivariance_from_numpy_array(point_support: np.ndarray,
                                                     variogram_model: TheoreticalVariogram):
    """
    Method calculates inblock semivariance of a given areas from the numpy array.


    Parameters
    ----------
    point_support : numpy array
                    [block index, x, y, value]

    variogram_model : TheoreticalVariogram
                      Modeled variogram fitted to the areal data.

    Returns
    -------
    inblock_semivariances : Dict
                            {area id: the average inblock semivariance}
    """

    inblock_semivariances = {}
    unique_areas = np.unique(point_support[:, 0])

    for unique_area in unique_areas:
        data_points = point_support[point_support[:, 0] == unique_area]

        data_points = data_points[:, 1:]

        inblock = inblock_semivariance(data_points, variogram_model)
        inblock_semivariances[unique_area] = inblock

    return inblock_semivariances


def _calculate_inblock_semivariance_from_point_support_class(point_support: PointSupport,
                                                             variogram_model: TheoreticalVariogram):
    """
    Method calculates inblock semivariance of a given areas from the PointSupport object.


    Parameters
    ----------
    point_support : PointSupport
                    Point support data.

    variogram_model : TheoreticalVariogram
                      Modeled variogram fitted to the areal data.

    Returns
    -------
    inblock_semivariances : Dict
                            {area id: the average inblock semivariance}
    """

    inblock_semivariances = {}
    unique_areas = (point_support.point_support[point_support.block_index_column]).unique()

    for unique_area in unique_areas:
        data_points = point_support.point_support[
            point_support.point_support[point_support.block_index_column] == unique_area
            ]

        data_points = data_points[
            [point_support.x_col, point_support.y_col, point_support.value_column]
        ].values

        inblock = inblock_semivariance(data_points, variogram_model)
        inblock_semivariances[unique_area] = inblock

    return inblock_semivariances
