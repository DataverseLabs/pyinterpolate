import numpy as np

from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.semivariogram.deconvolution.regularize import Deconvolution
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from .sample_data.dataprep import (CANCER_DATA_WITH_CENTROIDS,
                                   POINT_SUPPORT_DATA,
                                   T_DATA_BLOCKS, T_DATA_PS)


BLOCKS = Blocks(**CANCER_DATA_WITH_CENTROIDS)

PS = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=BLOCKS,
        points_value_column=POINT_SUPPORT_DATA['value_column_name'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column_name']
    )

MAX_RANGE = 300000
STEP_SIZE = 20000


def test_fit_and_transform_methods():
    dcv = Deconvolution(verbose=False)
    dcv.fit(blocks=BLOCKS,
            point_support=PS,
            nugget=0.0,
            step_size=STEP_SIZE,
            max_range=MAX_RANGE)

    assert isinstance(dcv._initial_theoretical_model, TheoreticalVariogram)
    assert dcv._s2 > 0
    assert isinstance(dcv._initial_theoretical_model_prediction, np.ndarray)
    assert dcv._initial_regularized_variogram is not None

    dcv.transform(max_iters=1)
    # dcv.export_model('regularized.json')
    assert isinstance(dcv.final_theoretical_model, TheoreticalVariogram)
    assert isinstance(dcv.final_regularized_variogram, np.ndarray)
    assert len(dcv.deviation.deviations) > 1


def test_fit_and_transform_methods_str_index():
    blocks_ds = Blocks(**T_DATA_BLOCKS)
    ps_ds = PointSupport(
        points=T_DATA_PS['ps'],
        blocks=blocks_ds,
        points_value_column=T_DATA_PS['value_column_name'],
        points_geometry_column=T_DATA_PS['geometry_column_name']
    )

    max_range = 90000
    step_size = 7500
    dcv = Deconvolution(verbose=False)
    dcv.fit(blocks=blocks_ds,
            point_support=ps_ds,
            nugget=0.0,
            step_size=step_size,
            max_range=max_range)

    assert isinstance(dcv._initial_theoretical_model, TheoreticalVariogram)
    assert dcv._s2 > 0
    assert isinstance(dcv._initial_theoretical_model_prediction, np.ndarray)
    assert dcv._initial_regularized_variogram is not None

    dcv.transform(max_iters=1)
    # dcv.export_model('regularized.json')
    assert isinstance(dcv.final_theoretical_model, TheoreticalVariogram)
    assert isinstance(dcv.final_regularized_variogram, np.ndarray)
    assert len(dcv.deviation.deviations) > 1

