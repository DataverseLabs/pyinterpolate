import unittest
import geopandas as gpd
import numpy

from pyinterpolate.processing.structure import get_polyset_from_file, get_polyset_from_geodataframe
from pyinterpolate.processing.utils.exceptions import WrongGeometryTypeError


class TestPolyset(unittest.TestCase):

    def test_read_data_from_shapefile(self):
        SHAPEFILE = '../samples/areal_data/test_areas_pyinterpolate.shp'
        polyset = get_polyset_from_file(SHAPEFILE, value_col='value', index_col='idx')
        self.assertTrue(polyset)


    def test_read_data_from_geojson(self):
        GEOJSON = '../samples/areal_data/test_areas_pyinterpolate.geojson'
        polyset = get_polyset_from_file(GEOJSON, value_col='value', index_col='idx')
        self.assertTrue(polyset)


    def test_read_data_from_geodataframe(self):
        GEOJSON = '../samples/areal_data/test_areas_pyinterpolate.geojson'
        gdf = gpd.read_file(GEOJSON)
        polyset = get_polyset_from_geodataframe(gdf, 'value', use_index=False)
        self.assertTrue(polyset)


    def test_keys_and_values(self):
        GEOJSON = '../samples/areal_data/test_areas_pyinterpolate.geojson'
        gdf = gpd.read_file(GEOJSON)
        polyset = get_polyset_from_geodataframe(gdf, 'value', use_index=True)

        # Test keys
        kk = {'points', 'igeom', 'info'}
        check_keys = set(polyset.keys()) == kk
        self.assertTrue(check_keys)

        # Test points
        self.assertIsInstance(polyset['points'], numpy.ndarray)

        # Test geometries
        self.assertIsInstance(polyset['igeom'], numpy.ndarray)

        # Test info
        info_keys = {'index_name', 'geom_name', 'val_name', 'crs'}
        check_info_keys = set(polyset['info'].keys()) == info_keys
        self.assertTrue(check_info_keys)

    def test_wrong_geometry(self):
        SHAPEFILE = '../samples/point_data/shapefile/test_points_pyinterpolate.shp'
        gdf = gpd.read_file(SHAPEFILE)
        with self.assertRaises(WrongGeometryTypeError):
            _ = get_polyset_from_geodataframe(gdf, value_col='value', index_col='id')
