from typing import Dict

import numpy as np
import pandas as pd

import geopandas as gpd

from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.core.data_models.point_support_distances import PointSupportDistance
from .sample_data.dataprep import CANCER_DATA_WITH_CENTROIDS, POINT_SUPPORT_DATA


def test_simple_case():
    blocks = Blocks(**CANCER_DATA_WITH_CENTROIDS)

    ps = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=blocks,
        points_value_column=POINT_SUPPORT_DATA['value_column_name'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column_name']
    )

    psd = PointSupportDistance()

    assert psd.weighted_block_to_block_distances is None
    assert not psd.distances_between_point_supports
    assert psd.no_closest_neighbors == 0
    assert not psd.closest_neighbors

    psd.calculate_weighted_block_to_block_distances(ps)
    assert isinstance(psd.weighted_block_to_block_distances, pd.DataFrame)
    assert len(psd.weighted_block_to_block_distances) > 0

    indexes = ps.blocks.block_indexes
    idx = np.random.choice(indexes)
    psd.calculate_point_support_distances(point_support=ps,
                                          block_id=idx,
                                          no_closest_neighbors=3)
    assert psd.no_closest_neighbors == 3
    print(psd.closest_neighbors)
    print(psd.distances_between_point_supports)
    print(psd._block_indexes)
    print(psd._calculated_block_pairs)
    assert isinstance(psd.closest_neighbors, Dict)

    weighted_distace = psd.get_weighted_distance(idx)
    assert isinstance(weighted_distace, pd.Series)
