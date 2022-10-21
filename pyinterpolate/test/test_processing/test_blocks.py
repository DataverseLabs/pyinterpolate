import unittest

import geopandas as gpd

from pyinterpolate.processing.preprocessing.blocks import Blocks, PointSupport


SHAPEFILE = 'samples/areal_data/test_areas_pyinterpolate.shp'
GEOJSON = 'samples/areal_data/test_areas_pyinterpolate.geojson'
SHAPEFILE_PTS = 'samples/point_data/shapefile/test_points_pyinterpolate.shp'

DATASET = 'samples/regularization/cancer_data.gpkg'
POLYGON_LAYER = 'areas'
POPULATION_LAYER = 'points'
POP10 = 'POP10'
GEOMETRY_COL = 'geometry'
POLYGON_ID = 'FIPS'


class TestPolyset(unittest.TestCase):

    def test_read_data_from_shapefile(self):
        blocks = Blocks()
        blocks.from_file(SHAPEFILE, value_col='value', index_col='idx')
        ds = blocks.data
        self.assertTrue(not ds.empty)

    def test_read_data_from_geojson(self):
        blocks = Blocks()
        blocks.from_file(GEOJSON, value_col='value', index_col='idx')
        ds = blocks.data
        self.assertTrue(not ds.empty)

    def test_read_data_from_geodataframe(self):
        gdf = gpd.read_file(GEOJSON)
        blocks = Blocks()
        blocks.from_geodataframe(gdf, value_col='value', geometry_col='geometry', index_col='idx')
        ds = blocks.data
        self.assertTrue(not ds.empty)

    def test_keys_and_values(self):
        gdf = gpd.read_file(GEOJSON)
        blocks = Blocks()
        blocks.from_geodataframe(gdf, value_col='value', geometry_col='geometry')
        expected_columns = {'index', 'value', 'geometry', 'centroid_x', 'centroid_y'}
        columns = set(blocks.data.columns)
        self.assertEqual(expected_columns, columns)


class TestPointSupportDataClass(unittest.TestCase):

    def test_get_from_files_fn(self):
        ps = PointSupport()
        ps.from_files(DATASET,
                      DATASET,
                      point_support_geometry_col=GEOMETRY_COL,
                      point_support_val_col=POP10,
                      blocks_geometry_col=GEOMETRY_COL,
                      blocks_index_col=POLYGON_ID,
                      point_support_layer_name=POPULATION_LAYER,
                      blocks_layer_name=POLYGON_LAYER)
        self.assertTrue(not ps.point_support.empty)
        expected_keys = {POP10, GEOMETRY_COL, POLYGON_ID, 'x_col', 'y_col'}
        out_keys = set(ps.point_support.keys())
        self.assertEqual(expected_keys, out_keys)

    def test_get_from_geodataframes_fn(self):
        gdf_points = gpd.read_file(DATASET, layer=POPULATION_LAYER)
        gdf_polygons = gpd.read_file(DATASET, layer=POLYGON_LAYER)
        ps = PointSupport()
        ps.from_geodataframes(gdf_points,
                              gdf_polygons,
                              point_support_geometry_col=GEOMETRY_COL,
                              point_support_val_col=POP10,
                              blocks_geometry_col=GEOMETRY_COL,
                              blocks_index_col=POLYGON_ID)
        self.assertTrue(not ps.point_support.empty)
        expected_keys = {POP10, GEOMETRY_COL, POLYGON_ID, 'x_col', 'y_col'}
        out_keys = set(ps.point_support.keys())
        self.assertEqual(expected_keys, out_keys)

