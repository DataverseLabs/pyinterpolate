import numpy as np
import pandas as pd

from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from pyinterpolate.distance.block import calc_block_to_block_distance
from .sample_data.dataprep import CANCER_DATA_WITH_CENTROIDS, POINT_SUPPORT_DATA


BLOCKS = Blocks(**CANCER_DATA_WITH_CENTROIDS)

PS = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=BLOCKS,
        points_value_column=POINT_SUPPORT_DATA['value_column_name'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column_name']
    )


def test_block_to_block_distance():
    distances = calc_block_to_block_distance(PS)
    distances_from_gdf = calc_block_to_block_distance(PS.point_support,
                                                      lon_col_name=PS.lon_col_name,
                                                      lat_col_name=PS.lat_col_name,
                                                      val_col_name=PS.value_column_name,
                                                      block_id_col_name=PS.point_support_blocks_index_name)
    assert isinstance(distances, pd.DataFrame)
    for k in distances.index:
        assert k in PS.unique_blocks
    for k, v in distances_from_gdf.items():
        assert np.sum(v) == np.sum(distances[k])
