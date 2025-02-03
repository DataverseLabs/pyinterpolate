from typing import Dict

from pyinterpolate import ExperimentalVariogram
from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.kriging.block.area_to_point_poisson_kriging import area_to_point_pk
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
        step_size=40000,
        max_range=300001
    )

THEO = TheoreticalVariogram()
THEO.autofit(
    experimental_variogram=EXPERIMENTAL,
    sill=150
)


EXPERIMENTAL_DIR = ExperimentalVariogram(
    ds=BLOCKS.representative_points_array(),
    step_size=40000,
    max_range=300001,
    direction=15,
    tolerance=0.2
)
THEO_DIR = TheoreticalVariogram()
THEO_DIR.autofit(
    experimental_variogram=EXPERIMENTAL_DIR,
    sill=150
)


THEO_PARAMS = {"experimental_variogram": None,
               "nugget": 0.0,
               "sill": 180,
               "rang": 180000,
               "variogram_model_type": "linear",
               "direction": None,
               "spatial_dependence": None,
               "spatial_index": None,
               "yhat": None,
               "errors": None}
THEO_FROM_REG = TheoreticalVariogram()
THEO_FROM_REG.from_dict(THEO_PARAMS)


def test_atp():
    indexes = BLOCKS.block_indexes

    atp_pk = area_to_point_pk(
        semivariogram_model=THEO_FROM_REG,
        point_support=PS,
        unknown_block_index=indexes[-1],
        number_of_neighbors=16
    )
    assert isinstance(atp_pk, Dict)


def test_ata_directional():
    indexes = BLOCKS.block_indexes

    atp = area_to_point_pk(
        semivariogram_model=THEO_DIR,
        point_support=PS,
        unknown_block_index=indexes[-1],
        number_of_neighbors=8
    )
    assert isinstance(atp, Dict)
