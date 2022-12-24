import unittest
from typing import Dict

from pyinterpolate.processing.preprocessing.blocks import PointSupport, Blocks
from pyinterpolate.variogram import build_experimental_variogram, TheoreticalVariogram
from pyinterpolate.variogram.regularization.block.inblock_semivariance import calculate_inblock_semivariance

DATASET = 'samples/regularization/cancer_data.gpkg'
POLYGON_LAYER = 'areas'
POPULATION_LAYER = 'points'
POP10 = 'POP10'
GEOMETRY_COL = 'geometry'
POLYGON_ID = 'FIPS'
POLYGON_VALUE = 'rate'
MAX_RANGE = 400000
STEP_SIZE = 40000

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

POINT_SUPPORT_GEODATAFRAME_RAW = POINT_SUPPORT_INPUT.point_support.copy()
POINT_SUPPORT_GEODATAFRAME = POINT_SUPPORT_GEODATAFRAME_RAW.rename(columns={
    POLYGON_ID: 'index',
    POP10: 'ds'
})

POINT_SUPPORT_NP_ARRAY = POINT_SUPPORT_GEODATAFRAME[['index', 'x_col', 'y_col', 'ds']].values

bc = AREAL_INPUT.data[[AREAL_INPUT.cx, AREAL_INPUT.cy, AREAL_INPUT.value_column_name]].values
experimental_variogram_of_areal_data = build_experimental_variogram(bc,
                                                                    step_size=STEP_SIZE,
                                                                    max_range=MAX_RANGE)

theoretical_model = TheoreticalVariogram()
theoretical_model.autofit(experimental_variogram_of_areal_data,
                          number_of_ranges=64,
                          number_of_sills=64,
                          deviation_weighting='closest')


class TestDeconvolution(unittest.TestCase):

    def test_point_support_input(self):
        inblock_semivariances = calculate_inblock_semivariance(POINT_SUPPORT_INPUT,
                                                               variogram_model=theoretical_model)
        self.assertTrue(inblock_semivariances)
        self.assertIsInstance(inblock_semivariances, Dict)
        self.assertEqual(
            set(
                inblock_semivariances.keys()
            ),
            set(
                POINT_SUPPORT_INPUT.point_support[POINT_SUPPORT_INPUT.block_index_column].unique()
            )
        )

    def test_geodataframe_input(self):
        inblock_semivariances = calculate_inblock_semivariance(POINT_SUPPORT_GEODATAFRAME,
                                                               variogram_model=theoretical_model)
        self.assertTrue(inblock_semivariances)
        self.assertIsInstance(inblock_semivariances, Dict)
        self.assertEqual(
            set(
                inblock_semivariances.keys()
            ),
            set(
                POINT_SUPPORT_INPUT.point_support[POINT_SUPPORT_INPUT.block_index_column].unique()
            )
        )

    def test_wrong_geodataframe_input(self):
        self.assertRaises(KeyError, calculate_inblock_semivariance, POINT_SUPPORT_GEODATAFRAME_RAW, theoretical_model)

    def test_numpy_input(self):
        inblock_semivariances = calculate_inblock_semivariance(POINT_SUPPORT_NP_ARRAY,
                                                               variogram_model=theoretical_model)
        self.assertTrue(inblock_semivariances)
        self.assertIsInstance(inblock_semivariances, Dict)
        self.assertEqual(
            set(
                inblock_semivariances.keys()
            ),
            set(
                POINT_SUPPORT_INPUT.point_support[POINT_SUPPORT_INPUT.block_index_column].unique()
            )
        )
