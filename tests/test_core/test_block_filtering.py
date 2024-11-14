import geopandas as gpd

from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.core.pipelines.block_filter import filter_blocks
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from tests.test_semivariogram.sample_data.dataprep import CANCER_DATA_WITH_CENTROIDS, POINT_SUPPORT_DATA


BLOCKS = Blocks(**CANCER_DATA_WITH_CENTROIDS)

PS = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=BLOCKS,
        points_value_column=POINT_SUPPORT_DATA['value_column'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column']
    )

THEO_PARAMS = {"experimental_variogram": None,
               "nugget": 0.0,
               "sill": 180,
               "rang": 200000,
               "variogram_model_type": "linear",
               "direction": None,
               "spatial_dependence": None,
               "spatial_index": None,
               "yhat": None,
               "errors": None}
THEO_FROM_REG = TheoreticalVariogram()
THEO_FROM_REG.from_dict(THEO_PARAMS)

def test_cpk():
    filtered = filter_blocks(
        semivariogram_model=THEO_FROM_REG,
        point_support=PS,
        number_of_neighbors=8,
        kriging_type='cb',
        verbose=True,
        raise_when_negative_error=False
    )
    assert isinstance(filtered, gpd.GeoDataFrame)


def test_ata():
    filtered = filter_blocks(
        semivariogram_model=THEO_FROM_REG,
        point_support=PS,
        number_of_neighbors=8,
        kriging_type='ata',
        verbose=True,
        raise_when_negative_error=False
    )
    assert isinstance(filtered, gpd.GeoDataFrame)


def test_atp():
    filtered = filter_blocks(
        semivariogram_model=THEO_FROM_REG,
        point_support=PS,
        number_of_neighbors=8,
        kriging_type='atp',
        verbose=True,
        raise_when_negative_error=False
    )
    assert isinstance(filtered, gpd.GeoDataFrame)
