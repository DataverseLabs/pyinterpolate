from pyinterpolate import ExperimentalVariogram
from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.kriging.block.centroid_based_poisson_kriging import centroid_poisson_kriging
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from tests.test_semivariogram.sample_data.dataprep import CANCER_DATA_WITH_CENTROIDS, POINT_SUPPORT_DATA


BLOCKS = Blocks(**CANCER_DATA_WITH_CENTROIDS)

PS = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=BLOCKS,
        points_value_column=POINT_SUPPORT_DATA['value_column'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column']
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
    sill=150,
    models_group='linear'
)


THEO_PARAMS = {"experimental_variogram": None,
               "nugget": 0.0,
               "sill": 176.3272376248095,
               "rang": 180000,
               "variogram_model_type": "spherical",
               "direction": None,
               "spatial_dependence": None,
               "spatial_index": None,
               "yhat": None,
               "errors": None}
THEO_FROM_REG = TheoreticalVariogram()
THEO_FROM_REG.from_dict(THEO_PARAMS)


def test_cpk():
    indexes = BLOCKS.block_indexes

    cpk = centroid_poisson_kriging(
        semivariogram_model=THEO_FROM_REG,
        point_support=PS,
        unknown_block_index=indexes[-1],
        number_of_neighbors=8
    )
    assert isinstance(cpk, dict)
    assert cpk['block_id'] == indexes[-1]
    assert cpk['zhat'] > 0
    assert cpk['sig'] > 0


def test_cpk_non_weighted():
    indexes = BLOCKS.block_indexes

    cpk = centroid_poisson_kriging(
        semivariogram_model=THEO,
        point_support=PS,
        unknown_block_index=indexes[-1],
        number_of_neighbors=8,
        is_weighted_by_point_support=False
    )
    assert isinstance(cpk, dict)
    assert cpk['block_id'] == indexes[-1]
    assert cpk['zhat'] > 0
    assert cpk['sig'] > 0
    assert True


def test_cpk_directional_non_weighted():
    indexes = BLOCKS.block_indexes

    cpk = centroid_poisson_kriging(
        semivariogram_model=THEO_DIR,
        point_support=PS,
        unknown_block_index=indexes[-1],
        number_of_neighbors=4,
        is_weighted_by_point_support=False,
        raise_when_negative_error=False,
        raise_when_negative_prediction=True
    )
    assert isinstance(cpk, dict)
    assert cpk['block_id'] == indexes[-1]
    assert cpk['zhat'] > 0
    assert True
