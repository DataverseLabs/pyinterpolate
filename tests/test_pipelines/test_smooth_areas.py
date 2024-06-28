import unittest
import geopandas as gpd
from pyinterpolate import smooth_blocks, Blocks, PointSupport, TheoreticalVariogram


DATASET = '../samples/regularization/cancer_data.gpkg'
VARIOGRAM_MODEL_FILE = '../samples/regularization/regularized_variogram.json'
POLYGON_LAYER = 'areas'
POPULATION_LAYER = 'points'
POP10 = 'POP10'
GEOMETRY_COL = 'geometry'
POLYGON_ID = 'FIPS'
POLYGON_VALUE = 'rate'
NN = 8

AREAL_INPUT = Blocks()
AREAL_INPUT.from_file(DATASET, value_col=POLYGON_VALUE, index_col=POLYGON_ID, layer_name=POLYGON_LAYER)
POINT_SUPPORT_INPUT = PointSupport()
POINT_SUPPORT_INPUT.from_files(point_support_data_file=DATASET,
                               blocks_file=DATASET,
                               point_support_geometry_col=GEOMETRY_COL,
                               point_support_val_col=POP10,
                               blocks_geometry_col=GEOMETRY_COL,
                               blocks_index_col=POLYGON_ID,
                               use_point_support_crs=True,
                               point_support_layer_name=POPULATION_LAYER,
                               blocks_layer_name=POLYGON_LAYER)

THEORETICAL_VARIOGRAM = TheoreticalVariogram()
THEORETICAL_VARIOGRAM.from_json(VARIOGRAM_MODEL_FILE)
DIRECTIONAL_VARIOGRAM_MODEL_FILE = '../samples/regularization/regularized_directional_variogram.json'
DIRECTIONAL_VARIOGRAM = TheoreticalVariogram()
DIRECTIONAL_VARIOGRAM.from_json(DIRECTIONAL_VARIOGRAM_MODEL_FILE)


class TestSmoothBlocks(unittest.TestCase):

    def test_real_data(self):

        output_results = smooth_blocks(semivariogram_model=THEORETICAL_VARIOGRAM,
                                       blocks=AREAL_INPUT,
                                       point_support=POINT_SUPPORT_INPUT,
                                       number_of_neighbors=NN)

        self.assertIsInstance(output_results, gpd.GeoDataFrame)

    def test_directional(self):
        output_results = smooth_blocks(semivariogram_model=DIRECTIONAL_VARIOGRAM,
                                       blocks=AREAL_INPUT,
                                       point_support=POINT_SUPPORT_INPUT,
                                       number_of_neighbors=NN)

        self.assertIsInstance(output_results, gpd.GeoDataFrame)

    def test_blocks_as_geodataframe(self):
        blocks_gdf = gpd.read_file(
            DATASET, layer=POLYGON_LAYER
        )
        # ``centroid.x, centroid.y, ds, index`` - must be in geodataframe
        blocks_gdf['centroid'] = blocks_gdf.centroid
        blocks_gdf['centroid.x'] = blocks_gdf['centroid'].x
        blocks_gdf['centroid.y'] = blocks_gdf['centroid'].y
        blocks_gdf.rename(columns={
            'rate': 'ds',
            'FIPS': 'index'
        }, inplace=True)

        smoothed = smooth_blocks(
            semivariogram_model=THEORETICAL_VARIOGRAM,
            blocks=blocks_gdf,
            point_support=POINT_SUPPORT_INPUT,
            number_of_neighbors=NN
        )
        self.assertIsInstance(smoothed, gpd.GeoDataFrame)
