from multiprocessing import Manager, Pool
from typing import Dict, Tuple, Union

import geopandas as gpd
import numpy as np

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


    # TODO: part below to test with very large datasets
    # unique_distances, uniq_count = np.unique(distances_between_points, return_counts=True)  # Array is flattened here
    # semivariances = variogram_model.predict(unique_distances)
    # multiplied_semivariances = semivariances * uniq_count

    average_block_semivariance = np.sum(semivariances) / p
    return average_block_semivariance


def calculate_inblock_semivariance(point_support: Union[PointSupport, gpd.GeoDataFrame, np.ndarray],
                                   variogram_model: TheoreticalVariogram) -> Dict:
    """
    Method calculates inblock semivariance of a given areas.


    Parameters
    ----------
    point_support : geopandas GeoDataFrame | Point Support | numpy array
                    Point support data. It can be provided:
                        - directly as PointSupport object,
                        - GeoDataFrame (then GeoDataFrame must have columns: 'ds' - values, 'x' - point
                          geometry x, 'y' - point geometry y, 'index' - block indexes,
                        - numpy array [block index, coordinate x, coordinate y, value].

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

    inblock_semivariances = {}

    # Use func for different types of input

    if isinstance(point_support, PointSupport):
        pass
    elif isinstance(point_support, gpd.GeoDataFrame):
        pass
    elif isinstance(point_support, np.ndarray):
        pass
    else:
        raise TypeError(f'Point support type {type(point_support)} not recognized. You may use PointSupport,'
                        f' Geopandas GeoDataFrame or numpy array. See docs.')

    return inblock_semivariances.copy()


def _calculate_inblock_semivariance_from_point_support_class(point_support, variogram_model):
    block_indexes =