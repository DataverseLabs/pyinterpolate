import unittest
import geopandas as gpd

from pyinterpolate.processing.point.structure import get_point_support_from_geodataframes,\
    get_point_support_from_files,\
    PointSupportDataClass


DATASET = 'samples/regularization/cancer_data.gpkg'
POLYGON_LAYER = 'areas'
POPULATION_LAYER = 'points'
POP10 = 'POP10'
GEOMETRY_COL = 'geometry'
POLYGON_ID = 'FIPS'


class TestPointSupportDataClass(unittest.TestCase):

    def test_get_from_files_fn(self):
        out = get_point_support_from_files(DATASET,
                                           DATASET,
                                           point_support_geometry_col=GEOMETRY_COL,
                                           point_support_val_col=POP10,
                                           polygon_geometry_col=GEOMETRY_COL,
                                           polygon_index_col=POLYGON_ID,
                                           point_support_layer_name=POPULATION_LAYER,
                                           polygon_layer_name=POLYGON_LAYER)
        self._test_cases(out)

    def test_get_from_geodataframes_fn(self):
        gdf_points = gpd.read_file(DATASET, layer=POPULATION_LAYER)
        gdf_polygons = gpd.read_file(DATASET, layer=POLYGON_LAYER)
        out = get_point_support_from_geodataframes(gdf_points, gdf_polygons,
                                                   point_support_geometry_col=GEOMETRY_COL,
                                                   point_support_val_col=POP10,
                                                   polygon_geometry_col=GEOMETRY_COL,
                                                   polygon_index_col=POLYGON_ID)
        self._test_cases(out)

    def test_load_from_files(self):
        point_support = PointSupportDataClass()
        out = point_support.from_files(DATASET,
                                       DATASET,
                                       point_support_geometry_col=GEOMETRY_COL,
                                       point_support_val_col=POP10,
                                       polygon_geometry_col=GEOMETRY_COL,
                                       polygon_index_col=POLYGON_ID,
                                       point_support_layer_name=POPULATION_LAYER,
                                       polygon_layer_name=POLYGON_LAYER)

        self._test_cases(out)

    def test_load_from_geodataframes(self):
        gdf_points = gpd.read_file(DATASET, layer=POPULATION_LAYER)
        gdf_polygons = gpd.read_file(DATASET, layer=POLYGON_LAYER)
        point_support = PointSupportDataClass()
        out = point_support.from_geodataframes(gdf_points,
                                               gdf_polygons,
                                               point_support_geometry_col=GEOMETRY_COL,
                                               point_support_val_col=POP10,
                                               polygon_geometry_col=GEOMETRY_COL,
                                               polygon_index_col=POLYGON_ID)
        self._test_cases(out)

    def _test_cases(self, output):
        # Check if all keys are present
        df = gpd.read_file(DATASET, layer=POLYGON_LAYER)
        uniq_keys = set(df[POLYGON_ID].unique())

        out_keys = set(output['data'].keys())
        msg = 'All areas present in the input should be present in the output file.'
        self.assertEqual(uniq_keys, out_keys, msg=msg)

        # Check the size of output arrays
        expected_size = 3

        for _key in list(out_keys):
            _size = output['data'][_key].shape[1]
            msg = f'Output data size should be (*, 3) but it is (*, {_size})'
            self.assertEqual(expected_size, _size, msg=msg)
