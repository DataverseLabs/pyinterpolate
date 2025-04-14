import random

import numpy as np
import pandas as pd
import pytest

from pyinterpolate.semivariogram.experimental.classes.experimental_variogram import ExperimentalVariogram
from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.semivariogram.theoretical.classes.theoretical_variogram import TheoreticalVariogram
from pyinterpolate.transform.select_poisson_kriging_data import select_poisson_kriging_data
from tests.test_semivariogram.sample_data.dataprep import CANCER_DATA_WITH_CENTROIDS, POINT_SUPPORT_DATA


BLOCKS = Blocks(**CANCER_DATA_WITH_CENTROIDS)

PS = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=BLOCKS,
        points_value_column=POINT_SUPPORT_DATA['value_column_name'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column_name']
    )

MAX_RANGE = 300000
STEP_SIZE = 39000

EXP = ExperimentalVariogram(
            ds=BLOCKS.representative_points_array(),
            step_size=STEP_SIZE,
            max_range=MAX_RANGE,
            direction=15,
            tolerance=0.2
        )

THEO = TheoreticalVariogram()
THEO.autofit(
    experimental_variogram=EXP,
)


def test_select_centroid_pk_data():
    indexes = BLOCKS.block_indexes
    idx = random.choice(indexes)
    result = select_poisson_kriging_data(
        block_index=idx,
        point_support=PS,
        semivariogram_model=THEO,
        number_of_neighbors=8,
        neighbors_range=THEO._study_max_range
    )

    assert isinstance(result.kriging_input, pd.DataFrame)

    weighted_semis = result.weighted_b2b_semivariance()
    assert isinstance(weighted_semis, pd.Series)

