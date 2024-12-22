import numpy as np
import pytest

from pyinterpolate import ExperimentalVariogram
from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.semivariogram.deconvolution.aggregated_variogram import AggregatedVariogram, regularize
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


def test_aggregated_variogram_class():
    agg_var = AggregatedVariogram(
        blocks=BLOCKS,
        point_support=PS,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        verbose=True
    )

    # Check if raise AttributeError without regularization
    with pytest.raises(AttributeError):
        agg_var.show_semivariograms()

    _ = agg_var.regularize()

    expected_lags = np.arange(STEP_SIZE, MAX_RANGE, STEP_SIZE)
    assert np.array_equal(expected_lags, agg_var.agg_lags)
    assert np.all(agg_var.regularized_variogram[:, 1] >= 0)


def test_regularize_fn():
    variogram = regularize(
        blocks=BLOCKS,
        point_support=PS,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        verbose=True
    )

    expected_lags = np.arange(STEP_SIZE, MAX_RANGE, STEP_SIZE)
    assert np.array_equal(expected_lags, variogram[:, 0])
    assert np.all(variogram[:, 1] >= 0)


def test_compare_cls_to_fn():
    agg_var = AggregatedVariogram(
        blocks=BLOCKS,
        point_support=PS,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        verbose=True
    )
    cls_variogram = agg_var.regularize()
    fn_variogram = regularize(
        blocks=BLOCKS,
        point_support=PS,
        step_size=STEP_SIZE,
        max_range=MAX_RANGE,
        verbose=True
    )
    assert np.array_equal(cls_variogram, fn_variogram)
