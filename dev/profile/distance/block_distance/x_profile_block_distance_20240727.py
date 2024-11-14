"""
Status: done, calculations 2x faster, no threading or parallelization
"""

import numpy as np
import matplotlib.pyplot as plt

from time import perf_counter

from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.distance.block import calc_block_to_block_distance, calc_block_to_block_distance_2
from dev.profile.distance.block_distance.dataprep import CANCER_DATA_WITH_CENTROIDS, POINT_SUPPORT_DATA


if __name__ == '__main__':
    BLOCKS = Blocks(**CANCER_DATA_WITH_CENTROIDS)

    PS = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=BLOCKS,
        points_value_column=POINT_SUPPORT_DATA['value_column'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column']
    )

    # mean t = 91
    # median.t = 90
    # std.t = 1.8
    times = []
    for i in range(5):
        print('*'*i)
        start = perf_counter()
        distances = calc_block_to_block_distance(PS)
        dt = perf_counter() - start
        times.append(dt)

    print('mean t')
    print(np.mean(times))
    print('median t')
    print(np.median(times))
    print('std t')
    print(np.std(times))

    vals = []
    for val in distances.values():
        vals.append(val)

    arr = np.array(vals)
    print('is tr')
    print(np.sum(np.triu(arr)), np.sum(np.tril(arr)))

    # plt.figure()
    # plt.imshow(arr, cmap='Reds')
    # plt.show()

    times = []
    for i in range(5):
        print('*'*i)
        start = perf_counter()
        distances = calc_block_to_block_distance_2(PS)
        dt = perf_counter() - start
        times.append(dt)

    # mean t = 45.63
    # median.t = 45.64
    # std.t = 0.34
    print('mean t 2')
    print(np.mean(times))
    print('median t 2')
    print(np.median(times))
    print('std t 2')
    print(np.std(times))

    vals = []
    for val in distances.values():
        vals.append(val)

    arr2 = np.array(vals)
    print(np.allclose(arr, arr2))
    print(np.sum(arr), np.sum(arr2))
