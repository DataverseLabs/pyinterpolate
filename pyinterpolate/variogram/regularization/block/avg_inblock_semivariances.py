"""
Functions for calculating the inblock semivariances.

Authors
-------
1. Szymon Moliński | @SimonMolinsky
"""
from typing import Dict

import numpy as np

from pyinterpolate.processing.select_values import select_values_in_range


def group_distances(block_to_block_distances: Dict, lags: np.ndarray, step_size: float) -> Dict:
    """
    Function prepares lag-neighbor-blocks Dict for semivariance calculations.

    Parameters
    ----------
    block_to_block_distances : Dict
                               {block id: [distances to all blocks in an order of dict ids]}

    lags : numpy array
           Array with lags.

    step_size : float

    Returns
    -------
    grouped_lags : Dict
                   {lag: {area id: [list of neighbors within a lag]}}
    """

    grouped_lags = {}

    block_ids = np.array(list(block_to_block_distances.keys()))

    for lag in lags:
        grouped_lags[lag] = {}
        for block_name, block in block_to_block_distances.items():
            distances_in_range = select_values_in_range(block, lag, step_size)
            if len(distances_in_range[0]) > 0:
                grouped_lags[lag][block_name] = block_ids[distances_in_range[0]]

    return grouped_lags


def calculate_average_semivariance(block_to_block_distances: Dict,
                                   inblock_semivariances: Dict,
                                   block_step_size: float,
                                   block_max_range: float) -> np.ndarray:
    """
    Function calculates average inblock semivariance between blocks.

    Parameters
    ----------
    block_to_block_distances : Dict
                               {block id : [distances to other blocks in order of keys]}

    inblock_semivariances : Dict
                            {area id: the inblock semivariance}

    block_step_size : float
                      Step size between lags.

    block_max_range : float
                      Maximal distance of analysis.

    Returns
    -------
    avg_block_to_block_semivariance : numpy array
                                      [lag, semivariance, number of blocks within lag]


    Notes
    -----
    Average inblock semivariance between blocks is defined as:

    $$\gamma_{h}(v, v) = \frac{1}{2*N(h)} \sum_{a=1}^{N(h)} \gamma(v_{a}, v_{a}) + \gamma(v_{a_h}, v_{a_h})$$

    where:
        - $\gamma_{h}(v, v)$ - average inblock semivariance per lag,
        - $N(h)$ - number of block pairs within a lag,
        - $\gamma(v_{a}, v_{a})$ - inblock semivariance of block a,
        - $\gamma(v_{a_h}, v_{a_h})$ - inblock semivariance of neighbouring block at a distance h.
    """

    avg_block_to_block_semivariance = []

    # Create lags
    lags = np.arange(block_step_size, block_max_range, block_step_size)

    # Select distances
    block_distances_per_lag = group_distances(block_to_block_distances, lags, block_step_size)

    # Calculate average semivariance per lag
    for lag in lags:
        average_semivariance = []
        number_of_blocks_per_lag = []
        for block_name, block_neighbors in block_distances_per_lag[lag].items():
            no_of_areas = len(block_neighbors)
            if no_of_areas > 0:
                partial_neighbors = [x for x in block_neighbors if x != block_name]
                n_len = len(partial_neighbors)
                if n_len > 0:
                    n_semivariances = [inblock_semivariances[bid] for bid in partial_neighbors]
                    average_semivariance.extend(n_semivariances)
                    number_of_blocks_per_lag.append(no_of_areas)

        # Average semivariance
        if len(average_semivariance) > 0:
            avg_semi = np.mean(average_semivariance) / 2
            pairs = np.sum(number_of_blocks_per_lag) / 2
            avg_block_to_block_semivariance.append([lag, avg_semi, pairs])
        else:
            avg_block_to_block_semivariance.append([lag, 0, 0])

    avg_block_to_block_semivariance = np.array(avg_block_to_block_semivariance)

    return avg_block_to_block_semivariance
