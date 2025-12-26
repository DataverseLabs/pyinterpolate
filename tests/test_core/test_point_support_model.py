import numpy as np
import pandas as pd

import geopandas as gpd

from shapely.geometry import Point, Polygon

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


def test_simple_case_sep_geoms():
    blocks = Blocks(**CANCER_DATA_WITH_CENTROIDS)

    values = POINT_SUPPORT_DATA['ps'][
        POINT_SUPPORT_DATA['value_column_name']
    ].values
    geometries = POINT_SUPPORT_DATA['ps'][
        POINT_SUPPORT_DATA['geometry_column_name']
    ]

    ps = PointSupport(
        blocks=blocks,
        values=values,
        geometries=geometries
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


def test_missing_block_values():
    block_values = [10, np.nan, 11]

    block_geoms = [
        Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        ),
        Polygon(
            [(10, 0), (20, 0), (20, 10), (10, 10), (10, 0)]
        ),
        Polygon(
            [(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)]
        )
    ]

    blocks = Blocks(
        values=block_values,
        geometries=block_geoms
    )

    points_values = [1, 2, 1, 3, 1, 5, 1, 6, 7, 7]
    points_geoms = [
        Point(2, 2),
        Point(3, 3),
        Point(4, 4),
        Point(15, 15),
        Point(16, 16),
        Point(17, 17),
        Point(18, 8),
        Point(19, 9),
        Point(10, 10),
        Point(12, 12)
    ]

    point_support = PointSupport(
        blocks=blocks,
        values=points_values,
        geometries=points_geoms
    )

    assert isinstance(point_support, PointSupport)
    assert isinstance(point_support.point_support, gpd.GeoDataFrame)

    points = point_support.get_points_array()
    assert isinstance(points, np.ndarray)

    indexes = point_support.get_point_to_block_indexes()
    assert isinstance(indexes, pd.Series)
    assert (len(indexes) == len(points))


def test_missing_block_geometries_none():
    block_values = [10, 10, 11]

    block_geoms = [
        Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        ),
        None,
        Polygon(
            [(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)]
        )
    ]

    blocks = Blocks(
        values=block_values,
        geometries=block_geoms
    )

    points_values = [1, 2, 1, 3, 1, 5, 1, 6, 7, 7]
    points_geoms = [
        Point(2, 2),
        Point(3, 3),
        Point(4, 4),
        Point(15, 15),
        Point(16, 16),
        Point(17, 17),
        Point(18, 8),
        Point(19, 9),
        Point(10, 10),
        Point(12, 12)
    ]

    point_support = PointSupport(
        blocks=blocks,
        values=points_values,
        geometries=points_geoms
    )

    assert isinstance(point_support, PointSupport)
    assert isinstance(point_support.point_support, gpd.GeoDataFrame)

    points = point_support.get_points_array()
    assert isinstance(points, np.ndarray)

    indexes = point_support.get_point_to_block_indexes()
    assert isinstance(indexes, pd.Series)
    assert (len(indexes) == len(points))


def test_missing_block_geometries_nan():
    block_values = [10, 10, 11]

    block_geoms = [
        Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        ),
        np.nan,
        Polygon(
            [(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)]
        )
    ]

    blocks = Blocks(
        values=block_values,
        geometries=block_geoms
    )

    points_values = [1, 2, 1, 3, 1, 5, 1, 6, 7, 7]
    points_geoms = [
        Point(2, 2),
        Point(3, 3),
        Point(4, 4),
        Point(15, 15),
        Point(16, 16),
        Point(17, 17),
        Point(18, 8),
        Point(19, 9),
        Point(10, 10),
        Point(12, 12)
    ]

    point_support = PointSupport(
        blocks=blocks,
        values=points_values,
        geometries=points_geoms
    )

    assert isinstance(point_support, PointSupport)
    assert isinstance(point_support.point_support, gpd.GeoDataFrame)

    points = point_support.get_points_array()
    assert isinstance(points, np.ndarray)

    indexes = point_support.get_point_to_block_indexes()
    assert isinstance(indexes, pd.Series)
    assert (len(indexes) == len(points))


def test_missing_point_support_values():
    block_values = [10, 9, 11]

    block_geoms = [
        Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        ),
        Polygon(
            [(10, 0), (20, 0), (20, 10), (10, 10), (10, 0)]
        ),
        Polygon(
            [(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)]
        )
    ]

    blocks = Blocks(
        values=block_values,
        geometries=block_geoms
    )

    points_values = [1, 2, 1, np.nan, 1, 5, None, 6, 7, 7]
    points_geoms = [
        Point(2, 2),
        Point(3, 3),
        Point(4, 4),
        Point(15, 15),
        Point(16, 16),
        Point(17, 17),
        Point(18, 8),
        Point(19, 9),
        Point(10, 10),
        Point(12, 12)
    ]

    point_support = PointSupport(
        blocks=blocks,
        values=points_values,
        geometries=points_geoms
    )

    assert isinstance(point_support, PointSupport)
    assert isinstance(point_support.point_support, gpd.GeoDataFrame)

    points = point_support.get_points_array()
    assert isinstance(points, np.ndarray)

    indexes = point_support.get_point_to_block_indexes()
    assert isinstance(indexes, pd.Series)
    assert (len(indexes) == len(points))


def test_missing_point_support_geometries():
    block_values = [10, 9, 11]

    block_geoms = [
        Polygon(
            [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        ),
        Polygon(
            [(10, 0), (20, 0), (20, 10), (10, 10), (10, 0)]
        ),
        Polygon(
            [(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)]
        )
    ]

    blocks = Blocks(
        values=block_values,
        geometries=block_geoms
    )

    points_values = [1, 2, 1, 3, 1, 5, 1, 6, 7, 7]
    points_geoms = [
        Point(2, 2),
        Point(3, 3),
        Point(4, 4),
        None,
        Point(16, 16),
        Point(17, 17),
        np.nan,
        Point(19, 9),
        Point(10, 10),
        Point(12, 12)
    ]

    point_support = PointSupport(
        blocks=blocks,
        values=points_values,
        geometries=points_geoms
    )

    assert isinstance(point_support, PointSupport)
    assert isinstance(point_support.point_support, gpd.GeoDataFrame)

    points = point_support.get_points_array()
    assert isinstance(points, np.ndarray)

    indexes = point_support.get_point_to_block_indexes()
    assert isinstance(indexes, pd.Series)
    assert (len(indexes) == len(points))

