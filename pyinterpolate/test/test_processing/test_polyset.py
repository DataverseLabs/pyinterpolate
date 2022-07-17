import unittest
from typing import Dict

import geopandas as gpd
import numpy as np

from pyinterpolate.processing.polygon.structure import get_polyset_from_file, get_polyset_from_geodataframe
from pyinterpolate.processing.utils.exceptions import WrongGeometryTypeError


SHAPEFILE = 'samples/areal_data/test_areas_pyinterpolate.shp'
GEOJSON = 'samples/areal_data/test_areas_pyinterpolate.geojson'
SHAPEFILE_PTS = 'samples/point_data/shapefile/test_points_pyinterpolate.shp'


class TestPolyset(unittest.TestCase):

    def test_read_data_from_shapefile(self):
        polyset = get_polyset_from_file(SHAPEFILE, value_col='value', index_col='idx')
        self.assertTrue(polyset)


    def test_read_data_from_geojson(self):
        polyset = get_polyset_from_file(GEOJSON, value_col='value', index_col='idx')
        self.assertTrue(polyset)


    def test_read_data_from_geodataframe(self):
        gdf = gpd.read_file(GEOJSON)
        polyset = get_polyset_from_geodataframe(gdf, 'value', use_index=False)
        self.assertTrue(polyset)


    def test_keys_and_values(self):
        gdf = gpd.read_file(GEOJSON)
        polyset = get_polyset_from_geodataframe(gdf, 'value', use_index=True)

        # Test keys
        kk = {'geometry', 'info', 'data'}
        check_keys = set(polyset.keys()) == kk
        self.assertTrue(check_keys)

        # Test instances
        self.assertIsInstance(polyset['geometry'], Dict)
        self.assertIsInstance(polyset['data'], np.ndarray)

        # Test info
        info_keys = {'index_name', 'geometry_name', 'value_name', 'crs'}
        check_info_keys = set(polyset['info'].keys()) == info_keys
        self.assertTrue(check_info_keys)

    def test_wrong_geometry(self):
        gdf = gpd.read_file(SHAPEFILE_PTS)
        with self.assertRaises(WrongGeometryTypeError):
            _ = get_polyset_from_geodataframe(gdf, value_col='value', index_col='id')
