from typing import Dict, Tuple

import numpy as np

from scipy.spatial.distance import cdist


def calc_block_to_block_distance(blocks: Dict) -> Tuple:
    """
    Function calculates distances between blocks.

    Parameters
    ----------
    blocks : Dict
             {block id: [[point x, point y, value]]}

    Returns
    -------
    block_distances, block_ids : Tuple[Dict, Tuple]
                                 {block id : [distances to other blocks]}, (block ids in the order of distances)
    """
    block_keys = tuple(blocks.keys())
    block_distances = dict()
    for k_i in block_keys:
        i_block = blocks[k_i]
        distances = []
        for k_j in block_keys:
            j_block = blocks[k_j]
            if k_i == k_j:
                distances.append(0)
            else:
                value = _calculate_block_to_block_distance(i_block, j_block)
                distances.append(value)
        block_distances[k_i] = distances

    return block_distances, block_keys


def _calculate_block_to_block_distance(block_1: np.ndarray, block_2: np.ndarray) -> float:
    """
    Function calculates distance between two blocks based on how they are divided (into the point support grid).

    Parameters
    ----------
    block_1 : numpy array

    block_2 : numpy array

    Returns
    -------
    weighted_distances : float
                         Weighted distance between blocks.

    Notes
    -----
    The weighted distance between blocks is

    :param area_block_1: set of coordinates of each population block in the form [x, y, value],
    :param area_block_2: the same set of coordinates as area_block_1.
    :return distance: weighted array of block to block distance.
    Equation: Dist(v_a, v_b) = 1 / (SUM_to(Pa), SUM_to(Pb) n(u_s) * n(u_si)) *
        * SUM_to(Pa), SUM_to(Pb) n(u_s) * n(u_si) ||u_s - u_si||
    where:
    Pa and Pb: number of points u_s and u_si used to discretize the two units v_a and v_b
    n(u_s) - population size in the cell u_s
    """

    if isinstance(area_block_1, list):
        area_block_1 = np.array(area_block_1)

    if isinstance(area_block_2, list):
        area_block_2 = np.array(area_block_2)

    a_shape = area_block_1.shape[0]
    b_shape = area_block_2.shape[0]
    ax = area_block_1[:, 0].reshape(1, a_shape)
    bx = area_block_2[:, 0].reshape(b_shape, 1)
    dx = ax - bx
    ay = area_block_1[:, 1].reshape(1, a_shape)
    by = area_block_2[:, 1].reshape(b_shape, 1)
    dy = ay - by
    aval = area_block_1[:, -1].reshape(1, a_shape)
    bval = area_block_2[:, -1].reshape(b_shape, 1)
    w = aval * bval

    dist = np.sqrt(dx ** 2 + dy ** 2)

    wdist = dist * w
    distances_sum = np.sum(wdist) / np.sum(w)
    return distances_sum


# TEMPORARY FUNCTIONS
def calc_point_to_point_distance(points_a, points_b=None):
    """temporary function for pt to pt distance estimation"""

    if points_b is None:
        distances = cdist(points_a, points_a, 'euclidean')
    else:
        distances = cdist(points_a, points_b, 'euclidean')
    return distances