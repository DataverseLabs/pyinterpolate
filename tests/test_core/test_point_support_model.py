import numpy as np
import pandas as pd

import geopandas as gpd

from pyinterpolate.core.data_models.blocks import Blocks
from pyinterpolate.core.data_models.point_support import PointSupport
from .sample_data.dataprep import CANCER_DATA_WITH_CENTROIDS, POINT_SUPPORT_DATA


def test_simple_case():
    blocks = Blocks(**CANCER_DATA_WITH_CENTROIDS)

    ps = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=blocks,
        points_value_column=POINT_SUPPORT_DATA['value_column_name'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column_name']
    )

    assert isinstance(ps, PointSupport)
    assert isinstance(ps.point_support, gpd.GeoDataFrame)

    points = ps.get_points_array()
    assert isinstance(points, np.ndarray)

    indexes = ps.get_point_to_block_indexes()
    assert isinstance(indexes, pd.Series)
    assert (len(indexes) == len(points))


def test_different_crs_case_1():
    blocks = Blocks(**CANCER_DATA_WITH_CENTROIDS)
    blocks.ds = blocks.ds.to_crs(epsg=3395)
    new_crs = blocks.ds.crs

    ps = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=blocks,
        points_value_column=POINT_SUPPORT_DATA['value_column_name'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column_name']
    )

    assert (ps.point_support.crs == new_crs)


def test_different_crs_case_2():
    blocks = Blocks(**CANCER_DATA_WITH_CENTROIDS)

    point_support_data = POINT_SUPPORT_DATA['ps']
    point_support_data.to_crs(epsg=3395)
    new_crs = point_support_data.crs

    ps = PointSupport(
        points=point_support_data,
        blocks=blocks,
        points_value_column=POINT_SUPPORT_DATA['value_column_name'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column_name'],
        use_point_support_crs=True
    )

    assert (ps.blocks.ds.crs == new_crs)


def test_stored_points():
    blocks = Blocks(**CANCER_DATA_WITH_CENTROIDS)

    ps = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=blocks,
        points_value_column=POINT_SUPPORT_DATA['value_column_name'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column_name'],
        store_dropped_points=True
    )

    assert len(ps.dropped_points) > 0


def test_unique_blocks():
    blocks = Blocks(**CANCER_DATA_WITH_CENTROIDS)

    ps = PointSupport(
        points=POINT_SUPPORT_DATA['ps'],
        blocks=blocks,
        points_value_column=POINT_SUPPORT_DATA['value_column_name'],
        points_geometry_column=POINT_SUPPORT_DATA['geometry_column_name'],
        store_dropped_points=True
    )

    assert len(ps.unique_blocks) == len(blocks.ds.index)
