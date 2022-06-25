from multiprocessing import Manager, Pool
from typing import Dict, Tuple

import numpy as np

from pyinterpolate.distance.distance import calc_point_to_point_distance
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


def multi_inblock_semivariance(output_dict: Dict, block: Tuple, variogram_model: TheoreticalVariogram):
    """
    Function calculates inblock semivariance and updates given dict with a results.

    Parameters
    ----------
    output_dict : Dict

    block : Tuple
            (area id, points array)

    variogram_model : TheoreticalVariogram

    """
    semi = inblock_semivariance(block[1], variogram_model)
    output_dict[block[0]] = semi


def calculate_inblock_semivariance(point_support: Dict,
                                   variogram_model: TheoreticalVariogram,
                                   n_workers=1) -> Dict:
    """
    Method calculates inblock semivariance of a given areas.


    Parameters
    ----------
    point_support : Dict
                    Point support data as a Dict:

                    point_support = {
                        'area_id': [numpy array with points]
                    }

    variogram_model : TheoreticalVariogram
                      Modeled variogram fitted to the areal data.

    n_workers : int, default = 1
                Set to > 1 to parallelize calculations.

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
    # Sequential version
    if n_workers == 1:
        for area in list(point_support.keys()):
            inblock_semi = inblock_semivariance(point_support[area], variogram_model)
            inblock_semivariances[area] = inblock_semi
    # Multiprocessing version
    else:
        manager = Manager()
        inblock_semivariances = manager.dict()
        pool = Pool(n_workers)

        for pointset in point_support.items():
            pool.apply_async(multi_inblock_semivariance, args=(inblock_semivariances, pointset, variogram_model))

        pool.close()
        pool.join()

    return inblock_semivariances.copy()
