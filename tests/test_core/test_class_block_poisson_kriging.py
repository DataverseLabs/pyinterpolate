import geopandas as gpd

from pyinterpolate import ExperimentalVariogram
from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.core.pipelines.block_filter import BlockPoissonKriging
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


def test_cases():
    cb_bpk = BlockPoissonKriging(
        semivariogram_model=THEO,
        point_support=PS,
        kriging_type='cb',
        verbose=True
    )

    ata_bpk = BlockPoissonKriging(
        semivariogram_model=THEO,
        point_support=PS,
        kriging_type='ata',
        verbose=True
    )

    atp_bpk = BlockPoissonKriging(
        semivariogram_model=THEO,
        point_support=PS,
        kriging_type='atp',
        verbose=True
    )

    nn = 8
    cb_reg = cb_bpk.regularize(
        number_of_neighbors=nn
    )
    ata_reg = ata_bpk.regularize(
        number_of_neighbors=nn
    )
    atp_reg = atp_bpk.regularize(
        number_of_neighbors=nn
    )

    print(cb_bpk.statistics)
    print(ata_bpk.statistics)
    print(atp_bpk.statistics)

    assert isinstance(cb_reg, gpd.GeoDataFrame)
    assert isinstance(ata_reg, gpd.GeoDataFrame)
    assert isinstance(atp_reg, gpd.GeoDataFrame)
