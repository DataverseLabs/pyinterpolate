from time import perf_counter

import numpy as np

from pyinterpolate import ExperimentalVariogram
from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.distance.block import calc_block_to_block_distance
from pyinterpolate.semivariogram.deconvolution.avg_inblock_semivariance import calculate_average_semivariance, calculate_average_semivariance_2
from pyinterpolate.semivariogram.deconvolution.inblock import calculate_inblock_semivariance
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from dev.profile.semivariogram.calculate_average_semivariance.dataprep import CANCER_DATA_WITH_CENTROIDS, POINT_SUPPORT_DATA


# Old
# def group_distances(block_to_block_distances: dict, lags: np.ndarray) -> dict:
#     """
#     Function prepares lag-neighbor-blocks Dict for semivariance calculations.
#
#     Parameters
#     ----------
#     block_to_block_distances : Dict
#                                {block id: [distances to all blocks in an order of dict ids]}
#
#     lags : numpy array
#
#     Returns
#     -------
#     grouped_lags : Dict
#                    {lag: {area id: [list of neighbors within a lag]}}
#     """
#
#     grouped_lags = {}
#
#     block_ids = np.array(list(block_to_block_distances.keys()))
#
#     for idx, _ in enumerate(lags):
#         current_lag, previous_lag = get_current_and_previous_lag(idx, lags)
#         grouped_lags[current_lag] = {}
#         for block_name, block in block_to_block_distances.items():
#             distances_in_range = select_values_in_range(block,
#                                                         current_lag=current_lag,
#                                                         previous_lag=previous_lag)
#             if len(distances_in_range[0]) > 0:
#                 grouped_lags[current_lag][block_name] = block_ids[distances_in_range[0]]
#
#     return grouped_lags
#
#
# def calculate_average_semivariance(block_to_block_distances: dict,
#                                    inblock_semivariances: dict,
#                                    step_size: float,
#                                    max_range: float) -> np.ndarray:
#     """
#     Function calculates average inblock semivariance between blocks.
#
#     Parameters
#     ----------
#     block_to_block_distances : Dict
#                                {block id : [distances to other blocks in order of keys]}
#
#     inblock_semivariances : Dict
#                             {area id: the inblock semivariance}
#
#     step_size : float
#                       Step size between lags.
#
#     max_range : float
#                       Maximal distance of analysis.
#
#     Returns
#     -------
#     avg_block_to_block_semivariance : numpy array
#                                       [lag, semivariance, number of blocks within lag]
#
#
#     Notes
#     -----
#     Average inblock semivariance between blocks is defined as:
#
#     $$\gamma_{h}(v, v) = \frac{1}{2*N(h)} \sum_{a=1}^{N(h)} \gamma(v_{a}, v_{a}) + \gamma(v_{a_h}, v_{a_h})$$
#
#     where:
#         - $\gamma_{h}(v, v)$ - average inblock semivariance per lag,
#         - $N(h)$ - number of block pairs within a lag,
#         - $\gamma(v_{a}, v_{a})$ - inblock semivariance of block a,
#         - $\gamma(v_{a_h}, v_{a_h})$ - inblock semivariance of neighbouring block at a distance h.
#     """
#
#     avg_block_to_block_semivariance = []
#
#     # Create lags
#     lags = np.arange(step_size, max_range, step_size)
#
#     # Select distances
#     block_distances_per_lag = group_distances(block_to_block_distances, lags)
#
#     # Calculate average semivariance per lag
#     for lag in lags:
#         average_semivariance = []
#         number_of_blocks_per_lag = []
#         for block_name, block_neighbors in block_distances_per_lag[lag].items():
#             no_of_areas = len(block_neighbors)
#             if no_of_areas > 0:
#                 partial_neighbors = [x for x in block_neighbors if x != block_name]
#                 n_len = len(partial_neighbors)
#                 if n_len > 0:
#                     n_semivariances = [inblock_semivariances[bid] for bid in partial_neighbors]
#                     average_semivariance.extend(n_semivariances)
#                     number_of_blocks_per_lag.append(no_of_areas)
#
#         # Average semivariance
#         if len(average_semivariance) > 0:
#             avg_semi = np.mean(average_semivariance) / 2
#             pairs = np.sum(number_of_blocks_per_lag) / 2
#             avg_block_to_block_semivariance.append([lag, avg_semi, pairs])
#         else:
#             avg_block_to_block_semivariance.append([lag, 0, 0])
#
#     avg_block_to_block_semivariance = np.array(avg_block_to_block_semivariance)
#
#     return avg_block_to_block_semivariance


BLOCKS = Blocks(**CANCER_DATA_WITH_CENTROIDS)

PS = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=BLOCKS,
        points_value_column=POINT_SUPPORT_DATA['value_column'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column']
    )

MAX_RANGE = 400000
STEP_SIZE = 20000

EXP = ExperimentalVariogram(
            ds=BLOCKS.representative_points_array(),
            step_size=STEP_SIZE,
            max_range=MAX_RANGE
        )

THEO = TheoreticalVariogram()
THEO.autofit(
    experimental_variogram=EXP,
    return_params=False
)

INBLOCK_SEMIVARS = calculate_inblock_semivariance(
        point_support=PS,
        variogram_model=THEO
    )

BLOCK_TO_BLOCK_DISTS = calc_block_to_block_distance(PS)


if __name__ == '__main__':
    times = []
    for i in range(5):
        print('*' * i)
        start = perf_counter()
        distances = calculate_average_semivariance(
            block_to_block_distances=BLOCK_TO_BLOCK_DISTS,
            inblock_semivariances=INBLOCK_SEMIVARS,
            step_size=STEP_SIZE,
            max_range=MAX_RANGE
        )
        dt = perf_counter() - start
        times.append(dt)

    print('mean t')
    print(np.mean(times))
    print('median t')
    print(np.median(times))
    print('std t')
    print(np.std(times))

    times = []
    for i in range(5):
        print('*' * i)
        start = perf_counter()
        distances_2 = calculate_average_semivariance_2(
            block_to_block_distances=BLOCK_TO_BLOCK_DISTS,
            inblock_semivariances=INBLOCK_SEMIVARS,
            step_size=STEP_SIZE,
            max_range=MAX_RANGE
        )
        dt = perf_counter() - start
        times.append(dt)

    print('mean t')
    print(np.mean(times))
    print('median t')
    print(np.median(times))
    print('std t')
    print(np.std(times))

    for idx, rec in enumerate(distances):
        print(rec)
        print(distances_2[idx])
