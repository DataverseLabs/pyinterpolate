from typing import Dict, Tuple, Collection

import numpy as np

from pyinterpolate.distance.distance import calc_point_to_point_distance
from pyinterpolate.variogram import TheoreticalVariogram


def block_pair_semivariance(block_a: Collection,
                            block_b: Collection,
                            semivariogram_model: TheoreticalVariogram):
    """
    Function calculates average semivariance between two blocks based on the counts inside the block.

    Parameters
    ----------
    block_a : Collection
              Block A points in the form of array with each record [x, y, value].

    block_b : Collection
              Block B points in the form of array with each record [x, y, value].

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


def calculate_centroid_block_to_block_semivariance(point_support: Dict,
                                                   block_to_block_distances: Tuple,
                                                   semivariogram_model: TheoreticalVariogram):
    """
    Function calculates semivariance between blocks based on their centroids and weighted distance between them.

    Parameters
    ----------
    point_support : Dict
                    Point support dict in the form:

                    point_support = {
                          'area_id': [numpy array with points and their values]
                    }

    block_to_block_distances : Tuple[Dict, Tuple]
                               {block id : [distances to other blocks]}, (block ids in the order of distances)

    semivariogram_model : TheoreticalVariogram
                          Fitted variogram model.

    Returns
    -------
    semivariances_b2b: Dict
                       {(block id a, block id b): [distance, semivariance, number of point pairs between blocks]}
    """

    blocks_ids = block_to_block_distances[1]
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
                    semivariances_b2b[pair] = semivariances_b2b[rev_pair].copy()
                else:
                    # Distance from one block to other
                    distance = block_to_block_distances[first_block_id][sec_indx]

                    # Select set of points to calculate block-pair semivariance
                    a_block_points = point_support[first_block_id][:, :-1]
                    b_block_points = point_support[second_block_id][:, :-1]
                    no_of_p_pairs = len(a_block_points) + len(b_block_points)

                    # Calculate semivariance between blocks
                    semivariance = block_pair_semivariance(a_block_points,
                                                           b_block_points,
                                                           semivariogram_model)

                    semivariances_b2b[pair] = [distance, semivariance, no_of_p_pairs]

    return semivariances_b2b
