# OPEN FOR DEVELOPMENT - threading made it slower *

from time import perf_counter

import numpy as np

from pyinterpolate import ExperimentalVariogram
from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.distance.block import calc_block_to_block_distance
from pyinterpolate.semivariogram.deconvolution.block_to_block_semivariance import calculate_block_to_block_semivariance
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from dev.profile.semivariogram.calculate_average_semivariance.dataprep import CANCER_DATA_WITH_CENTROIDS, POINT_SUPPORT_DATA


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

DISTANCES = calc_block_to_block_distance(PS)

if __name__ == '__main__':
    # 58.06809436399999
    start = perf_counter()
    distances = calculate_block_to_block_semivariance(
        PS,
        block_to_block_distances=DISTANCES,
        semivariogram_model=THEO
    )
    dt = perf_counter() - start
    print(dt)
