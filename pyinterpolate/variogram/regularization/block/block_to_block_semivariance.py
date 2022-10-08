"""
Functions for calculating the semivariances between blocks' point supports.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Dict, Collection, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from pyinterpolate.distance.distance import calc_point_to_point_distance
from pyinterpolate.processing.preprocessing.blocks import PointSupport
from pyinterpolate.processing.transform.transform import block_arr_to_dict, block_dataframe_to_dict, \
    point_support_to_dict
from pyinterpolate.variogram import TheoreticalVariogram


def block_pair_semivariance(block_a: Collection,
                            block_b: Collection,
                            semivariogram_model: TheoreticalVariogram):
    """
    Function calculates average semivariance between two blocks based on the counts inside the block.

    Parameters
    ----------
    block_a : Collection
              Block A points in the form of array with each record [x, y].

    block_b : Collection
              Block B points in the form of array with each record [x, y].

    semivariogram_model : TheoreticalVariogram
                          Fitted theoretical variogram model from TheoreticalVariogram class.

    Returns
    -------
    semivariance_between_blocks : float
                                  The average semivariance between blocks (calculated from point every point from
                                  block a to every point in block b).
    """

    distances_between_points = calc_point_to_point_distance(block_a, block_b).flatten()

    predictions = semivariogram_model.predict(distances_between_points)

    semivariance_between_blocks = np.mean(predictions)

    return semivariance_between_blocks


def _check_point_support(point_support: Union[Dict, PointSupport, gpd.GeoDataFrame, pd.DataFrame, np.ndarray]):
    if isinstance(point_support, np.ndarray):
        return block_arr_to_dict(point_support)
    elif isinstance(point_support, pd.DataFrame) or isinstance(point_support, gpd.GeoDataFrame):
        return block_dataframe_to_dict(point_support)
    elif isinstance(point_support, PointSupport):
        return point_support_to_dict(point_support)
    elif isinstance(point_support, Dict):
        return point_support
    else:
        raise TypeError(f'Unknown point support type {type(point_support)}. Expected types are: '
                        f'Dict, PointSupport, gpd.GeoDataFrame, pd.DataFrame, np.ndarray.')


def calculate_block_to_block_semivariance(
        point_support: Union[Dict, PointSupport, gpd.GeoDataFrame, pd.DataFrame, np.ndarray],
        block_to_block_distances: Dict,
        semivariogram_model: TheoreticalVariogram):
    """
    Function calculates semivariance between blocks based on their point support and weighted distance between
        block centroids.

    Parameters
    ----------
    point_support : Union[Dict, np.ndarray, gpd.GeoDataFrame, pd.DataFrame, PointSupport]
                    * Dict: {block id: [[point x, point y, value]]}
                    * numpy array: [[block id, x, y, value]]
                    * DataFrame and GeoDataFrame: columns={x, y, ds, index}
                    * PointSupport

    block_to_block_distances : Dict
                               {block id : [distances to other blocks in order of keys]}

    semivariogram_model : TheoreticalVariogram
                          Fitted variogram model.

    Returns
    -------
    semivariances_b2b: Dict
                       {(block id a, block id b): [distance, semivariance, number of point pairs between blocks]}
    """

    # Prepare data
    point_support = _check_point_support(point_support)

    # Run
    blocks_ids = list(block_to_block_distances.keys())
    semivariances_b2b = {}

    for first_block_id in blocks_ids:
        for sec_indx, second_block_id in enumerate(blocks_ids):

            pair = (first_block_id, second_block_id)
            rev_pair = (second_block_id, first_block_id)

            if first_block_id == second_block_id:
                semivariances_b2b[pair] = [0, 0, 0]
            else:
                if (pair not in semivariances_b2b) and (rev_pair in semivariances_b2b):
                    # Check if semivar is not actually calculated and skip calculations if it is
                    semivariances_b2b[pair] = semivariances_b2b[rev_pair]
                else:
                    # Distance from one block to other
                    distance = block_to_block_distances[first_block_id][sec_indx]

                    # Select set of points to calculate block-pair semivariance
                    a_block_points = point_support[first_block_id][:, :-1]
                    b_block_points = point_support[second_block_id][:, :-1]
                    # TODO: it was added, but should be multiplied - check other results
                    no_of_p_pairs = len(a_block_points) * len(b_block_points)

                    # Calculate semivariance between blocks
                    semivariance = block_pair_semivariance(a_block_points,
                                                           b_block_points,
                                                           semivariogram_model)

                    semivariances_b2b[pair] = [distance, semivariance, no_of_p_pairs]

    return semivariances_b2b
