import numpy as np

from pyinterpolate.semivariogram.experimental.classes.experimental_variogram import ExperimentalVariogram
from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.distance.block import calc_block_to_block_distance
from pyinterpolate.semivariogram.deconvolution.block_to_block_semivariance import calculate_block_to_block_semivariance, \
    average_block_to_block_semivariances
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from .sample_data.dataprep import CANCER_DATA_WITH_CENTROIDS, POINT_SUPPORT_DATA


BLOCKS = Blocks(**CANCER_DATA_WITH_CENTROIDS)

PS = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=BLOCKS,
        points_value_column=POINT_SUPPORT_DATA['value_column_name'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column_name']
    )

MAX_RANGE = 300000
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


def test_block_to_block_semivariance():
    b2b_sem = calculate_block_to_block_semivariance(
        point_support=PS,
        block_to_block_distances=DISTANCES,
        semivariogram_model=THEO
    )
    assert isinstance(b2b_sem, dict)
    assert len(b2b_sem) == len(PS.unique_blocks)**2


def test_average_block_to_block_semivariances():
    b2b_sem = calculate_block_to_block_semivariance(
        point_support=PS,
        block_to_block_distances=DISTANCES,
        semivariogram_model=THEO
    )

    semis = average_block_to_block_semivariances(
        semivariances_array=np.array(list(b2b_sem.values()), dtype=float),
        lags=THEO.lags
    )
    assert isinstance(semis, np.ndarray)
    assert len(semis) > 0
