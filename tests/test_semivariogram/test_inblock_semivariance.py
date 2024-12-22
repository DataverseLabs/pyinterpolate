from pyinterpolate import ExperimentalVariogram
from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.semivariogram.deconvolution.inblock import calculate_inblock_semivariance
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from .sample_data.dataprep import CANCER_DATA_WITH_CENTROIDS, POINT_SUPPORT_DATA


BLOCKS = Blocks(**CANCER_DATA_WITH_CENTROIDS)

PS = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=BLOCKS,
        points_value_column=POINT_SUPPORT_DATA['value_column_name'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column_name']
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

def test_inblock_semivariance():
    inblock_semivariances = calculate_inblock_semivariance(
        point_support=PS,
        variogram_model=THEO
    )
    assert isinstance(inblock_semivariances, dict)
    for k, v in inblock_semivariances.items():
        assert k in PS.unique_blocks
        assert v >= 0
