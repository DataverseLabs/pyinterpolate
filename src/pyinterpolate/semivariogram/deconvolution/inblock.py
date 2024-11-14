"""
Functions for calculating the inblock point-support semivariance.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Dict
import numpy as np

from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.distance.point import point_distance
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram




def inblock_semivariance(points_of_block: np.ndarray, variogram_model: TheoreticalVariogram) -> float:
    """
    Function calculates inblock semivariance.

    Parameters
    ----------
    points_of_block : numpy array
        ``[x, y, value]``

    variogram_model : TheoreticalVariogram

    Returns
    -------
    average : float

    """
    number_of_points_within_block = len(points_of_block)  # P
    p = number_of_points_within_block * number_of_points_within_block  # P^2

    distances_between_points: np.ndarray
    distances_between_points = point_distance(points_of_block[:, :-1],
                                              points_of_block[:, :-1])  # Matrix of size PxP
    flattened_distances = distances_between_points.flatten()

    semivariances = variogram_model.predict(flattened_distances)

    average = np.sum(semivariances) / p
    return average


def calculate_inblock_semivariance(point_support: PointSupport,
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

    inblock_semivariances = _calculate_inblock_semivariance_from_point_support_class(point_support,
                                                                                     variogram_model)

    return inblock_semivariances.copy()


def _calculate_inblock_semivariance_from_point_support_class(point_support: PointSupport,
                                                             variogram_model: TheoreticalVariogram) -> Dict:
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

    for unique_area in point_support.unique_blocks:
        ds = point_support.get_points_array(block_id=unique_area)
        inblock = inblock_semivariance(ds, variogram_model)
        inblock_semivariances[unique_area] = inblock

    return inblock_semivariances
