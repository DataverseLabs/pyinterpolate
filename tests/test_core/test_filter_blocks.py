import geopandas as gpd
import numpy as np

from pyinterpolate.semivariogram.experimental.classes.experimental_variogram import ExperimentalVariogram
from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.core.pipelines.block_filter import filter_blocks
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from tests.test_semivariogram.sample_data.dataprep import CANCER_DATA_WITH_CENTROIDS, POINT_SUPPORT_DATA


BLOCKS = Blocks(**CANCER_DATA_WITH_CENTROIDS)

PS = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=BLOCKS,
        points_value_column=POINT_SUPPORT_DATA['value_column_name'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column_name']
    )

EXPERIMENTAL = ExperimentalVariogram(
        ds=BLOCKS.representative_points_array(),
        step_size=30000,
        max_range=200001
    )

THEO = TheoreticalVariogram()
THEO.autofit(
    experimental_variogram=EXPERIMENTAL,
    sill=150
)


def test_case():
    filtered = filter_blocks(
        semivariogram_model=THEO,
        point_support=PS,
        number_of_neighbors=8,
        kriging_type='cb',
        verbose=True
    )
    assert isinstance(filtered, gpd.GeoDataFrame)

    # get block values
    max_val = np.max(PS.blocks.block_values)

    assert max_val > np.max(filtered['reg.est'].values)


