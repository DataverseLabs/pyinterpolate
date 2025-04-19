import numpy as np

from pyinterpolate.distance.block import select_neighbors_in_range
from pyinterpolate.semivariogram.lags.lags import get_current_and_previous_lag


def group_distances(block_to_block_distances: dict, lags: np.ndarray) -> dict:
    """
    Function prepares lag -> block -> distances to neighbors dictionary for
    semivariance calculations.

    Parameters
    ----------
    block_to_block_distances : Dict
        ``{block id: [distances to all blocks in an order of dict ids]}``

    lags : numpy array
        Ordered lags.

    Returns
    -------
    grouped_lags : Dict
        ``{lag: {area id: [list of neighbors within a lag]}}``
    """

    grouped_lags = {}

    for idx, _ in enumerate(lags):
        current_lag, previous_lag = get_current_and_previous_lag(idx, lags)
        neighbors_in_range = select_neighbors_in_range(
            block_to_block_distances,
            current_lag=current_lag,
            previous_lag=previous_lag
        )
        grouped_lags[current_lag] = neighbors_in_range

    return grouped_lags


def calculate_average_semivariance(block_to_block_distances: dict,
                                   inblock_semivariances: dict,
                                   step_size: float,
                                   max_range: float) -> np.ndarray:
    r"""
    Function calculates average inblock semivariance between blocks.

    Parameters
    ----------
    block_to_block_distances : Dict
        ``{block id : [distances to other blocks in order of keys]}``

    inblock_semivariances : Dict
        ``{area id: the inblock semivariance}``

    step_size : float
        Step size between lags.

    max_range : float
        Maximal distance of analysis.

    Returns
    -------
    avg_block_to_block_semivariance : numpy array
        ``[lag, semivariance, number of blocks within lag]``


    Notes
    -----
    Average inblock semivariance between blocks is defined as:

    $$\gamma_{h}(v, v) =
      \frac{1}{2*N(h)} \sum_{a=1}^{N(h)} \gamma(v_{a}, v_{a}) +
      \gamma(v_{a_h}, v_{a_h})$$

    where:
        - $\gamma_{h}(v, v)$ - average inblock semivariance per lag,
        - $N(h)$ - number of block pairs within a lag,
        - $\gamma(v_{a}, v_{a})$ - inblock semivariance of block a,
        - $\gamma(v_{a_h}, v_{a_h})$ - inblock semivariance of neighbouring
          block at a distance h.
    """

    avg_block_to_block_semivariance = []

    # Create lags
    lags = np.arange(step_size, max_range, step_size)

    # Select distances
    block_distances_per_lag = group_distances(block_to_block_distances, lags)

    # Calculate average semivariance per lag
    for lag in lags:
        neighbors = block_distances_per_lag[lag]
        no_of_areas = len(neighbors)
        if no_of_areas > 0:
            average_semivariance = [
                inblock_semivariances[block_id] for block_id in neighbors
            ]

            # Average semivariance
            if len(average_semivariance) > 0:
                avg_semi = np.mean(average_semivariance) / 2
                pairs = int(no_of_areas / 2)
                avg_block_to_block_semivariance.append([lag, avg_semi, pairs])
            else:
                avg_block_to_block_semivariance.append([lag, 0, 0])
        else:
            avg_block_to_block_semivariance.append([lag, 0, 0])

    avg_block_to_block_semivariance = np.array(avg_block_to_block_semivariance)

    return avg_block_to_block_semivariance
