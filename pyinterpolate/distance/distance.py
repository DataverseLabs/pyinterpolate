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
    The weighted distance between blocks is derived from the equation:

    $$d(v_{a}, v_{b})=\frac{1}{\sum_{s=1}^{P_{a}} \sum_{s'=1}^{P_{b}} n(u_{s}) n(u_{s'})} *
        \sum_{s=1}^{P_{a}} \sum_{s'=1}^{P_{b}} n(u_{s})n(u_{s'})||u_{s}-u_{s'}||$$

    where:
    $P_{a}$ and $P_{b}$: number of points $u_{s}$ and $u_{s'}$ used to discretize the two units $v_{a}$ and $v_{b}$,
    $n(u_{s})$ and $n(u_{s'})$ - population size in the cells $u_{s}$ and $u_{s'}$.

    References
    ----------
    .. [1] Goovaerts, P. Kriging and Semivariogram Deconvolution in the Presence of Irregular Geographical Units.
           Math Geosci 40, 101–128 (2008). https://doi.org/10.1007/s11004-007-9129-1
    """

    a_shape = block_1.shape[0]
    b_shape = block_2.shape[0]
    ax = block_1[:, 0].reshape(1, a_shape)
    bx = block_2[:, 0].reshape(b_shape, 1)
    dx = ax - bx
    ay = block_1[:, 1].reshape(1, a_shape)
    by = block_2[:, 1].reshape(b_shape, 1)
    dy = ay - by
    aval = block_1[:, -1].reshape(1, a_shape)
    bval = block_2[:, -1].reshape(b_shape, 1)
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
