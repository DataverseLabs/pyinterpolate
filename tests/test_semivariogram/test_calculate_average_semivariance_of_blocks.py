import numpy as np

from pyinterpolate import ExperimentalVariogram
from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.distance.block import calc_block_to_block_distance
from pyinterpolate.semivariogram.deconvolution.avg_inblock_semivariance import calculate_average_semivariance
from pyinterpolate.semivariogram.deconvolution.inblock import calculate_inblock_semivariance
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from .sample_data.dataprep import CANCER_DATA_WITH_CENTROIDS, POINT_SUPPORT_DATA


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

def test_avg_semi():
    avg_semi = calculate_average_semivariance(
        block_to_block_distances=BLOCK_TO_BLOCK_DISTS,
        inblock_semivariances=INBLOCK_SEMIVARS,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE
    )
    assert isinstance(avg_semi, np.ndarray)
